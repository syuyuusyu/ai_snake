"""Fine-tune stable_4 with sparse Jev guidance.

PPO stays fully on-policy (Jev never overrides the action during rollout).
Instead, critical states are recorded, labelled (cache -> rollouts -> Jev) and
used in an auxiliary behaviour-cloning loss after each PPO update (DAgger-style).

    .venv/bin/python train_with_jev.py                # real Jev
    .venv/bin/python train_with_jev.py --no-jev       # rollout labels only, 0 Jev calls
"""

import argparse
import copy
import random
import threading

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import train_with_ppo as base
from game_env import SnakeEnv
from jev_assist import (DIRS, GateState, JevCache, Labeler, LabelBuffer, Snapshot, canonical,
                        detect_gate, make_jev_fn, safe_mask_fn)

BOARD = 12


class SafeSnakeEnv(SnakeEnv):
    def mask_fn(self):
        return safe_mask_fn(self)


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
            self.pending.append(Snapshot.take(game, DIRS[int(action)], gate, self._obs, safe_mask_fn(self)))
        obs, r, term, trunc, info = self.env.step(action)
        if info.get("step_state") == "eat food":
            self.gs.on_eat()
        self._obs = obs
        return obs, r, term, trunc, info

    def pop_snapshots(self):
        out, self.pending = self.pending, []
        return out


def make_env(seed):
    def _init():
        env = SafeSnakeEnv(seed=seed, board_size=BOARD, silent_mode=True,
                           bfs_intensity=base.bfs_intensity, reward_gamma=base.discount_gamma)
        env = JevGateWrapper(env)
        env = ActionMasker(env, safe_mask_fn)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


class MaskablePPOWithBC(MaskablePPO):
    """MaskablePPO + auxiliary BC loss on labelled critical states."""
    bc_buffer: LabelBuffer | None = None
    bc_coef_start, bc_coef_end = 0.5, 0.05
    bc_steps, bc_batch, bc_min_size = 4, 256, 256

    def train(self):
        super().train()
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
            loss = coef * -(w * logp).sum() / w.sum()
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
    """Collects snapshots each rollout, labels them in a background thread so the
    PPO update is never blocked by network latency."""

    def __init__(self, labeler: Labeler, buffer: LabelBuffer, max_per_round=128):
        super().__init__()
        self.labeler, self.buffer, self.max_per_round = labeler, buffer, max_per_round
        self.thread, self.result = None, None

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self.thread is not None:
            self.thread.join()
            self.buffer.add(self.result or [])
            self.thread = None

    def _on_rollout_end(self):
        # harvest finished labels from the previous round
        if self.thread is not None and not self.thread.is_alive():
            self.buffer.add(self.result or [])
            self.thread = None
        snaps = [s for lst in self.training_env.env_method("pop_snapshots") for s in lst]
        if self.thread is None and snaps:
            # dedupe by canonical key, risk gates first
            uniq = {}
            for s in sorted(snaps, key=lambda s: s.gate != "risk"):
                uniq.setdefault(canonical(s.snake, s.food, BOARD)[0], s)
            batch = list(uniq.values())
            random.shuffle(batch)
            batch.sort(key=lambda s: s.gate != "risk")
            batch = batch[: self.max_per_round]
            # refresh the labeler's frozen CPU policy only between rounds
            self.labeler.policy.load_state_dict(
                {k: v.detach().cpu() for k, v in self.model.policy.state_dict().items()})

            def work():
                self.result = self.labeler.label(batch)
            self.thread = threading.Thread(target=work, daemon=True)
            self.thread.start()
        for k, v in self.labeler.stats.items():
            self.logger.record(f"jev/{k}", v)
        st = self.labeler.stats
        if st["audit"]:
            self.logger.record("jev/audit_agree_rate", st["audit_agree"] / st["audit"])


class SafeFixedSeedEval(base.FixedSeedEvalCallback):
    def _on_training_start(self):
        self.eval_env = SafeSnakeEnv(seed=self.seeds[0], board_size=BOARD, silent_mode=True,
                                     bfs_intensity=base.bfs_intensity, reward_gamma=base.discount_gamma)
        self.eval_policy = self.model.policy_class(
            self.model.observation_space, self.model.action_space, lambda _: 0.0,
            **self.model.policy_kwargs).to("cpu")
        self._evaluate_and_save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="pth/stable_4.zip")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--no-jev", action="store_true", help="rollout labels only")
    ap.add_argument("--jev-budget", type=int, default=32, help="max Jev calls per PPO round")
    ap.add_argument("--audit-rate", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--cache", default="evaluations/jev_cache.jsonl")
    ap.add_argument("--save", default="pth/stable_7_jev")
    args = ap.parse_args()

    seeds = random.sample(range(10_000_000), args.envs)
    env = SubprocVecEnv([make_env(s) for s in seeds])
    lr = base.schedule_fn(3e-6, 1e-6)
    clip = base.schedule_fn(0.08, 0.03)
    model = MaskablePPOWithBC.load(
        args.init, env=env, device=base.device,
        custom_objects={
            "observation_space": env.observation_space, "action_space": env.action_space,
            "learning_rate": lr, "lr_schedule": lr, "clip_range": clip,
            "gamma": base.discount_gamma, "gae_lambda": base.gae_lambda,
            "ent_coef": 0.003, "n_epochs": 2, "target_kl": 0.01,
            "n_steps": 512, "batch_size": 512 * 8,
        })
    for g in model.policy.optimizer.param_groups:
        g["lr"] = lr(1.0)

    buffer = LabelBuffer()
    model.bc_buffer = buffer
    cpu_policy = copy.deepcopy(model.policy).to("cpu")
    cpu_policy.set_training_mode(False)
    jev_fn = (lambda game: None) if args.no_jev else make_jev_fn()
    labeler = Labeler(
        make_env=lambda: SafeSnakeEnv(seed=0, board_size=BOARD, silent_mode=True,
                                      bfs_intensity=base.bfs_intensity),
        policy=cpu_policy, jev_fn=jev_fn, cache=JevCache(args.cache, BOARD),
        horizon=args.horizon, audit_rate=0.0 if args.no_jev else args.audit_rate,
        budget_per_round=0 if args.no_jev else args.jev_budget)

    callbacks = [
        base.MonitorCallback(),
        JevLabelCallback(labeler, buffer),
        SafeFixedSeedEval(eval_freq=max(500_000 // env.num_envs, 1), seeds=range(32),
                          best_model_path=f"{args.save}_best"),
    ]
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
        model.save(f"{args.save}_final")
    finally:
        env.close()
        print("labeler stats:", labeler.stats)


if __name__ == "__main__":
    main()
