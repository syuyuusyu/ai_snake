"""Fine-tune stable_4 with sparse Jev guidance.

PPO stays fully on-policy (Jev never overrides the action during rollout).
Instead, critical states are recorded, labelled (cache -> rollouts -> Jev) and
used in an auxiliary behaviour-cloning loss after each PPO update (DAgger-style).

    .venv/bin/python train_with_jev.py                # real Jev
    .venv/bin/python train_with_jev.py --no-jev       # rollout labels only, 0 Jev calls
    .venv/bin/python train_with_jev.py --agent claude # let `claude -p` tune hot parameters

With --agent, see agent_bridge.py and agent_supervision.md. A stop decision saves
<save>_restart.zip, writes <run-dir>/restart.json and exits with code 75 for supervisor.py.
"""

import argparse
import copy
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import train_with_ppo as base
from agent_bridge import (
    RESTART_EXIT_CODE,
    HOT_PARAMS,
    AgentBridgeCallback,
    HumanControlCallback,
    RunLog,
    ScaledSchedule,
    Triggers,
    Tunable,
    build_schema,
    make_claude_advisor,
    mock_advisor,
)
from board_config import (
    DEFAULT_BOARD_HIGH,
    DEFAULT_BOARD_WIDTH,
    add_board_arguments,
    validate_board_dimensions,
)
from game_env import SnakeEnv
from jev_assist import (
    DIRS,
    GateState,
    JevCache,
    LabelBuffer,
    Labeler,
    Shield,
    Snapshot,
    canonical,
    detect_gate,
    make_jev_fn,
)


class SafeSnakeEnv(SnakeEnv):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.shield = Shield(hunger=Shield.default_hunger(self.game))

    def reset(self, **kw):
        out = super().reset(**kw)
        self.shield.reset()
        self.shield.record(self.game, ate=False)
        return out

    def step(self, action):
        out = super().step(action)
        self.shield.record(self.game, ate=out[4]["step_state"] == "eat food")
        return out

    def mask_fn(self):
        return self.shield.mask(self.game)


class JevGateWrapper(gym.Wrapper):
    """Records critical states; never changes the action."""

    def __init__(self, env, max_per_rollout=8):
        super().__init__(env)
        self.gs, self.pending, self.max_per_rollout = GateState(), [], max_per_rollout
        self._obs = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._obs, self.gs = obs, GateState()
        return obs, info

    def step(self, action):
        game = self.env.unwrapped.game
        gate = detect_gate(game, DIRS[int(action)], self.gs)
        if gate and len(self.pending) < self.max_per_rollout and self._obs is not None:
            self.pending.append(Snapshot.take(game, DIRS[int(action)], gate, self._obs, self.env.unwrapped.mask_fn(),
                                              step_count=self.env.unwrapped.step_count))
        obs, r, term, trunc, info = self.env.step(action)
        if info.get("step_state") == "eat food":
            self.gs.on_eat()
        self._obs = obs
        return obs, r, term, trunc, info

    def pop_snapshots(self):
        out, self.pending = self.pending, []
        return out


def make_env(seed, board_width=DEFAULT_BOARD_WIDTH, board_high=DEFAULT_BOARD_HIGH, max_snapshots=8):
    def _init():
        env = SafeSnakeEnv(seed=seed, board_width=board_width, board_high=board_high, silent_mode=True,
                           bfs_intensity=base.bfs_intensity, reward_gamma=base.discount_gamma)
        env = JevGateWrapper(env, max_per_rollout=max_snapshots)
        env = ActionMasker(env, lambda e: e.unwrapped.mask_fn())
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


class MaskablePPOWithBC(MaskablePPO):
    """MaskablePPO + auxiliary BC loss on labelled critical states."""
    bc_buffer: LabelBuffer | None = None
    bc_coef_start, bc_coef_end = 0.5, 0.05
    bc_steps, bc_batch, bc_min_size = 4, 256, 256

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["bc_buffer"]

    def train(self):
        super().train()
        self.train_bc()

    def train_bc(self):
        buf = self.bc_buffer
        if buf is None or len(buf) < self.bc_min_size:
            return
        p = self._current_progress_remaining
        coef = self.bc_coef_end + p * (self.bc_coef_start - self.bc_coef_end)
        self.policy.set_training_mode(True)
        losses, accs = [], []
        for _ in range(self.bc_steps):
            obs, mask, act, w = buf.sample(self.bc_batch, self.device)
            _, logp, _ = self.policy.evaluate_actions(obs, act, action_masks=mask.cpu().numpy())
            loss = coef * -(w * logp).mean()
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            with torch.no_grad():
                dist = self.policy.get_distribution(obs, action_masks=mask.cpu().numpy())
                accs.append((dist.distribution.probs.argmax(-1) == act).float().mean().item())
            losses.append(loss.item())
        self.logger.record("bc/loss", float(np.mean(losses)))
        self.logger.record("bc/agree_with_label", float(np.mean(accs)))
        self.logger.record("bc/coef", coef)
        self.logger.record("bc/buffer", len(buf))


class JevLabelCallback(BaseCallback):
    """Label on a frozen CPU policy; harvest only on the training thread."""

    def __init__(self, labeler: Labeler, buffer: LabelBuffer, max_per_round=128, seed=0):
        super().__init__()
        self.labeler, self.buffer, self.max_per_round = labeler, buffer, max_per_round
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.rng = random.Random(seed)
        self.dropped_busy = 0

    def _on_step(self):
        return True

    def _harvest(self, wait=False):
        if self.future is None or (not wait and not self.future.done()):
            return False
        # Propagate programming/I/O errors instead of silently killing the label worker.
        labels = self.future.result()
        self.future = None
        self.buffer.add(labels)
        return bool(labels)

    def _on_training_end(self):
        try:
            if self._harvest(wait=True):
                self.model.train_bc()
            self._log_stats()
        finally:
            self.close()

    def close(self):
        self.pool.shutdown(wait=True)

    def _on_rollout_end(self):
        self._harvest()
        snaps = [s for lst in self.training_env.env_method("pop_snapshots") for s in lst]
        if self.future is not None:
            self.dropped_busy += len(snaps)
        elif snaps:
            game = self.labeler.env.game
            uniq = {}
            for snap in sorted(snaps, key=lambda s: s.gate != "risk"):
                key, _ = canonical(snap.snake, snap.food, game.board_width, game.board_high,
                                   snap.direction, snap.step_count)
                uniq.setdefault(key, snap)
            batch = list(uniq.values())
            self.rng.shuffle(batch)
            batch.sort(key=lambda s: s.gate != "risk")
            batch = batch[:self.max_per_round]
            self.labeler.policy.load_state_dict(
                {k: v.detach().cpu() for k, v in self.model.policy.state_dict().items()})
            self.labeler.policy.set_training_mode(False)
            self.future = self.pool.submit(self.labeler.label, batch)
        self._log_stats()

    def _log_stats(self):
        for k, v in self.labeler.stats.items():
            self.logger.record(f"jev/{k}", v)
        self.logger.record("jev/dropped_busy", self.dropped_busy)
        self.logger.record("bc/buffer", len(self.buffer))
        st = self.labeler.stats
        self.logger.record("jev/audit_agree_rate", st["audit_agree"] / max(st["audit"], 1))


class SafeFixedSeedEval(base.FixedSeedEvalCallback):
    def _on_training_start(self):
        self.eval_env = SafeSnakeEnv(seed=self.seeds[0], board_width=self.board_width,
                                     board_high=self.board_high, silent_mode=True,
                                     bfs_intensity=base.bfs_intensity, reward_gamma=base.discount_gamma)
        self.eval_policy = self.model.policy_class(
            self.model.observation_space, self.model.action_space, lambda _: 0.0,
            **self.model.policy_kwargs).to("cpu")
        self._evaluate_and_save()

    def _on_training_end(self):
        try:
            # Includes the last background labels applied by JevLabelCallback.
            self._evaluate_and_save()
        finally:
            super()._on_training_end()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_board_arguments(ap)
    ap.add_argument("--init", default="pth/stable_4.zip")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default=base.device)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-jev", action="store_true", help="rollout labels only, including cache hits")
    ap.add_argument("--jev-model", default="typesafe/jev-1.13")
    ap.add_argument("--jev-budget", type=int, default=32, help="max HTTP requests per label batch, audits included")
    ap.add_argument("--jev-workers", type=int, default=4)
    ap.add_argument("--audit-rate", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--max-labels", type=int, default=128)
    ap.add_argument("--snapshots-per-env", type=int, default=8)
    ap.add_argument("--bc-min-size", type=int, default=256)
    ap.add_argument("--bc-batch-size", type=int, default=256)
    ap.add_argument("--bc-steps", type=int, default=4)
    ap.add_argument("--eval-freq", type=int, default=500_000, help="environment steps; 0 disables evaluation")
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--eval-max-steps", type=int, default=20_000)
    ap.add_argument("--cache", type=Path, help="defaults to separate control/experiment caches")
    ap.add_argument("--save", type=Path, help="checkpoint prefix; default depends on --no-jev")
    ap.add_argument("--log-dir", type=Path, help="TensorBoard and CSV directory; defaults under logs/")
    ag = ap.add_argument_group("agent supervision (see agent_bridge.py)")
    ag.add_argument("--agent", choices=["off", "mock", "claude"], default="off",
                    help="who answers when an evaluation trips a rule")
    ag.add_argument("--agent-model", default="haiku")
    ag.add_argument("--agent-max-calls", type=int, default=10, help="advisor calls per run")
    ag.add_argument("--agent-max-usd", type=float, default=2.0, help="stop asking once total spend reaches this")
    ag.add_argument("--agent-call-usd", type=float, default=0.5, help="--max-budget-usd for each claude -p call")
    ag.add_argument("--agent-cooldown", type=int, default=2, help="evaluations to wait after each question")
    ag.add_argument("--agent-trigger", choices=["rules", "always"], default="rules",
                    help="'always' asks after every evaluation (dry runs only)")
    ag.add_argument("--claude-bin", default="claude")
    ag.add_argument("--run-dir", type=Path, default=Path("runs"), help="events/decisions/restart files")
    args = ap.parse_args(argv)
    try:
        validate_board_dimensions(args.board_width, args.board_high)
    except ValueError as exc:
        ap.error(str(exc))
    positive = ("envs", "timesteps", "n_steps", "batch_size", "jev_workers", "horizon", "max_labels",
                "snapshots_per_env", "bc_min_size", "bc_batch_size", "bc_steps", "eval_episodes", "eval_max_steps")
    if any(getattr(args, key) < 1 for key in positive):
        ap.error("training sizes, horizons and worker counts must be positive")
    if args.n_steps * args.envs < 2 or args.batch_size < 2:
        ap.error("rollout size and batch size must be >= 2")
    if args.jev_budget < 0 or args.eval_freq < 0 or not 0 <= args.audit_rate <= 1:
        ap.error("budgets and eval frequency must be nonnegative; audit rate must be in [0, 1]")
    if not 0 <= args.seed < 2**32 or args.envs > 10_000_000:
        ap.error("seed must be in [0, 2**32); envs must be <= 10000000")
    if args.agent != "off" and not args.eval_freq:
        ap.error("--agent needs evaluations; set --eval-freq > 0")
    if args.agent_max_calls < 0 or args.agent_max_usd < 0 or args.agent_cooldown < 0:
        ap.error("agent limits must be nonnegative")
    mode = "ctrl" if args.no_jev else "jev"
    args.save = args.save or Path(f"pth/stable_7_{mode}")
    args.cache = args.cache or Path(f"evaluations/{mode}_cache.jsonl")
    args.log_dir = args.log_dir or Path("logs") / args.save.name
    return args


def build_tunables(model, labeler, args) -> dict[str, Tunable]:
    """Hot parameters only (ranges in agent_bridge.HOT_PARAMS). Rollout sizes, gamma/lambda,
    rewards and the network need a restart or a code change."""
    bind = {
        "ent_coef": (lambda: model.ent_coef, lambda v: setattr(model, "ent_coef", v)),
        "target_kl": (lambda: model.target_kl, lambda v: setattr(model, "target_kl", v)),
        "lr_scale": (lambda: model.lr_schedule.scale, lambda v: setattr(model.lr_schedule, "scale", v)),
        "clip_scale": (lambda: model.clip_range.scale, lambda v: setattr(model.clip_range, "scale", v)),
        "bc_coef_start": (lambda: model.bc_coef_start, lambda v: setattr(model, "bc_coef_start", v)),
    }
    if not args.no_jev:
        bind["jev_budget"] = (lambda: labeler.budget, lambda v: setattr(labeler, "budget", int(v)))
        bind["audit_rate"] = (lambda: labeler.audit_rate, lambda v: setattr(labeler, "audit_rate", v))
    out = {}
    for name, (get, set_) in bind.items():
        lo, hi, integer, doc = HOT_PARAMS[name]
        out[name] = Tunable(name, lo, hi, get, set_, doc, integer)
    return out


def main(argv=None):
    args = parse_args(argv)
    set_random_seed(args.seed)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.log_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    seeds = random.Random(args.seed).sample(range(10_000_000), args.envs)
    factories = [make_env(s, args.board_width, args.board_high, args.snapshots_per_env) for s in seeds]
    env = DummyVecEnv(factories) if args.envs == 1 else SubprocVecEnv(factories)
    labeler = label_callback = eval_callback = bridge = None
    logger = None
    run_log = RunLog(args.run_dir)
    run_log.event("run_started", save=str(args.save), init=str(args.init), timesteps=args.timesteps,
                  agent=args.agent, no_jev=args.no_jev)
    exit_code = 0
    try:
        lr = base.schedule_fn(3e-6, 1e-6)
        clip = base.schedule_fn(0.08, 0.03)
        model = MaskablePPOWithBC.load(
            str(args.init), env=env, device=args.device,
            custom_objects={
                "observation_space": env.observation_space, "action_space": env.action_space,
                "learning_rate": lr, "lr_schedule": lr, "clip_range": clip,
                "gamma": base.discount_gamma, "gae_lambda": base.gae_lambda,
                "ent_coef": 0.003, "n_epochs": 2, "target_kl": 0.01,
                "n_steps": args.n_steps, "batch_size": min(args.batch_size, args.n_steps * args.envs),
                "seed": args.seed,
            })
        # Mutable multipliers so the agent can scale LR/clip; SB3 re-reads both every update.
        model.lr_schedule = ScaledSchedule(lr)
        model.clip_range = ScaledSchedule(clip)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = lr(1.0)
        logger = configure(str(args.log_dir), ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)
        buffer = LabelBuffer()
        model.bc_buffer = buffer
        model.bc_min_size, model.bc_batch, model.bc_steps = args.bc_min_size, args.bc_batch_size, args.bc_steps
        cpu_policy = copy.deepcopy(model.policy).to("cpu")
        cpu_policy.set_training_mode(False)
        labeler = Labeler(
            make_env=lambda: SafeSnakeEnv(seed=args.seed, board_width=args.board_width,
                                          board_high=args.board_high, silent_mode=True,
                                          bfs_intensity=base.bfs_intensity,
                                          reward_gamma=base.discount_gamma),
            policy=cpu_policy, jev_fn=None if args.no_jev else make_jev_fn(args.jev_model),
            cache=JevCache(args.cache, args.board_width, args.board_high),
            horizon=args.horizon, audit_rate=0.0 if args.no_jev else args.audit_rate,
            max_workers=args.jev_workers, budget_per_round=0 if args.no_jev else args.jev_budget,
            seed=args.seed)
        label_callback = JevLabelCallback(labeler, buffer, max_per_round=args.max_labels, seed=args.seed)
        callbacks = [base.MonitorCallback(), label_callback]
        if args.eval_freq:
            eval_callback = SafeFixedSeedEval(
                eval_freq=max(args.eval_freq // env.num_envs, 1), seeds=range(args.eval_episodes),
                max_episode_steps=args.eval_max_steps, best_model_path=f"{args.save}_best",
                board_width=args.board_width, board_high=args.board_high)
            callbacks.append(eval_callback)
        tunables = build_tunables(model, labeler, args)
        human = HumanControlCallback(tunables, run_log)   # train_ctl.py adjust/stop, always on
        callbacks.append(human)
        if args.agent != "off":
            advisor = mock_advisor if args.agent == "mock" else make_claude_advisor(
                build_schema(tunables), model=args.agent_model, max_budget_usd=args.agent_call_usd,
                claude_bin=args.claude_bin, run_dir=args.run_dir)
            bridge = AgentBridgeCallback(
                eval_callback, tunables, advisor, run_log, stats_fn=lambda: dict(labeler.stats),
                triggers=Triggers(always=args.agent_trigger == "always"),
                cooldown_evals=args.agent_cooldown, max_calls=args.agent_max_calls, max_usd=args.agent_max_usd)
            callbacks.append(bridge)
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
        if bridge is not None and bridge.restart_requested:
            ckpt = f"{args.save}_restart"
            model.save(ckpt)
            (args.run_dir / "restart.json").write_text(json.dumps({
                "checkpoint": f"{ckpt}.zip", "timesteps_done": int(model.num_timesteps),
                "timesteps_requested": args.timesteps, "reason": bridge.restart_reason}, indent=2) + "\n")
            run_log.event("restart_requested", checkpoint=f"{ckpt}.zip", timesteps_done=int(model.num_timesteps),
                          reason=bridge.restart_reason)
            exit_code = RESTART_EXIT_CODE
        else:
            model.save(f"{args.save}_final")
            stopped = bool(eval_callback is not None and eval_callback.stopped_for_degradation)
            run_log.event("run_finished", timesteps_done=int(model.num_timesteps), stopped_for_degradation=stopped,
                          stopped_by_human=human.stop_requested, stop_reason=human.stop_reason,
                          best_score=None if eval_callback is None else eval_callback.best_score)
        if logger.name_to_value:
            logger.dump(model.num_timesteps)
    except BaseException as exc:
        run_log.event("crash", error=repr(exc)[:500])
        raise
    finally:
        if label_callback is not None:
            label_callback.close()
        if labeler is not None:
            labeler.close()
            (args.log_dir / "jev_stats.json").write_text(json.dumps(labeler.stats, indent=2) + "\n")
            print("labeler stats:", labeler.stats)
        if eval_callback is not None and eval_callback.eval_env is not None:
            eval_callback.eval_env.close()
        env.close()
        if logger is not None:
            logger.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
