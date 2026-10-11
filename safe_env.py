"""Shielded snake env and the worker loop for WorkerVecEnv.

Nothing here may import torch, stable_baselines3 or sb3_contrib: WorkerVecEnv's
subprocesses import this module, and on Windows every process that imports the
CUDA build of torch commits ~1.3GB, so 64 env processes exceed the commit limit.
"""

import time

import gymnasium as gym

from board_config import DEFAULT_BOARD_HIGH, DEFAULT_BOARD_WIDTH
from game_env import SnakeEnv
from jev_assist import DIRS, GateState, Shield, Snapshot, detect_gate

# Same values as train_with_ppo.bfs_intensity / discount_gamma (that module imports torch).
BFS_INTENSITY = 1
DISCOUNT_GAMMA = 0.995


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


class TimeoutIsTerminal(gym.Wrapper):
    """Report the starvation timeout as terminated instead of truncated.

    SnakeEnv truncates after 4 * board area steps without food. SB3 treats truncation as
    a time limit and adds gamma * V(last obs) to the reward, which cancels the -1 timeout
    penalty, so the agent never learns that circling in the endgame loses the win.
    """

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if trunc and not term:
            term, trunc = True, False
        return obs, r, term, trunc, info


class MaskedMonitor(gym.Wrapper):
    """Torch-free stand-in for sb3 Monitor + sb3_contrib ActionMasker."""

    def __init__(self, env):
        super().__init__(env)
        self._ret, self._len, self._t0 = 0.0, 0, time.time()

    def action_masks(self):
        return self.env.unwrapped.mask_fn()

    def reset(self, **kw):
        self._ret, self._len = 0.0, 0
        return self.env.reset(**kw)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._ret += float(r)
        self._len += 1
        if term or trunc:
            info["episode"] = {"r": round(self._ret, 6), "l": self._len, "t": round(time.time() - self._t0, 6)}
        return obs, r, term, trunc, info


def build_env(seed, board_width=DEFAULT_BOARD_WIDTH, board_high=DEFAULT_BOARD_HIGH, labels=True, max_snapshots=8,
              timeout_terminal=False):
    env = SafeSnakeEnv(seed=seed, board_width=board_width, board_high=board_high, silent_mode=True,
                       bfs_intensity=BFS_INTENSITY, reward_gamma=DISCOUNT_GAMMA)
    if timeout_terminal:
        env = TimeoutIsTerminal(env)
    if labels:
        env = JevGateWrapper(env, max_per_rollout=max_snapshots)
    env = MaskedMonitor(env)
    env.reset(seed=seed)
    return env


def worker(remote, parent_remote, env_kwargs):
    """Runs several envs in one process; protocol matches WorkerVecEnv.

    step/reset also return each env's next action mask, saving MaskablePPO a round trip per step.
    """
    parent_remote.close()
    envs = [build_env(**kw) for kw in env_kwargs]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                out = []
                for env, action in zip(envs, data):
                    obs, r, term, trunc, info = env.step(action)
                    info["TimeLimit.truncated"] = trunc and not term
                    reset_info = {}
                    if term or trunc:
                        info["terminal_observation"] = obs
                        obs, reset_info = env.reset()
                    out.append((obs, r, term or trunc, info, reset_info, env.action_masks()))
                remote.send(out)
            elif cmd == "reset":
                out = []
                for env, (seed, opt) in zip(envs, data):
                    obs, reset_info = env.reset(seed=seed, **({"options": opt} if opt else {}))
                    out.append((obs, reset_info, env.action_masks()))
                remote.send(out)
            elif cmd == "env_method":
                local, name, args, kwargs = data
                remote.send([envs[i].get_wrapper_attr(name)(*args, **kwargs) for i in local])
            elif cmd == "get_attr":
                local, name = data
                remote.send([envs[i].get_wrapper_attr(name) for i in local])
            elif cmd == "set_attr":
                local, name, value = data
                for i in local:
                    envs[i].set_wrapper_attr(name, value)
                remote.send(None)
            elif cmd == "has_attr":
                try:
                    envs[0].get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "spaces":
                remote.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == "close":
                break
            else:
                raise NotImplementedError(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        for env in envs:
            env.close()
        remote.close()
