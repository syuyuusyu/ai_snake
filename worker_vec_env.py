"""VecEnv that runs several shielded envs per subprocess.

SubprocVecEnv starts one process per env, and each one imports torch through
stable_baselines3. With the CUDA build on Windows that is ~1.3GB of committed
memory per process. Workers here only import safe_env (no torch), and batching
envs per process also cuts pipe round-trips.
"""

import multiprocessing as mp
from itertools import pairwise

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import safe_env


class WorkerVecEnv(VecEnv):
    def __init__(self, env_kwargs: list[dict], n_workers: int):
        n_workers = max(1, min(n_workers, len(env_kwargs)))
        # Contiguous slices so env i always lives in the same worker.
        bounds = np.linspace(0, len(env_kwargs), n_workers + 1).astype(int)
        self.slices = [range(a, b) for a, b in pairwise(bounds)]
        ctx = mp.get_context("spawn")
        self.remotes, self.processes = [], []
        for sl in self.slices:
            remote, work_remote = ctx.Pipe()
            proc = ctx.Process(target=safe_env.worker, args=(work_remote, remote, [env_kwargs[i] for i in sl]),
                               daemon=True)
            proc.start()
            work_remote.close()
            self.remotes.append(remote)
            self.processes.append(proc)
        self.remotes[0].send(("spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        self.closed = False
        self._masks = None
        super().__init__(len(env_kwargs), observation_space, action_space)

    def _route(self, indices):
        """Group global env indices as {worker: [local index, ...]}."""
        out = {}
        for i in self._get_indices(indices):
            for w, sl in enumerate(self.slices):
                if i in sl:
                    out.setdefault(w, []).append(i - sl.start)
                    break
        return out

    def step_async(self, actions):
        for remote, sl in zip(self.remotes, self.slices):
            remote.send(("step", [int(a) for a in actions[sl.start:sl.stop]]))

    def step_wait(self):
        results = [r for remote in self.remotes for r in remote.recv()]
        obs, rews, dones, infos, self.reset_infos, self._masks = zip(*results)
        return np.stack(obs), np.array(rews, dtype=np.float32), np.array(dones), list(infos)

    def reset(self):
        for remote, sl in zip(self.remotes, self.slices):
            remote.send(("reset", [(self._seeds[i], self._options[i]) for i in sl]))
        results = [r for remote in self.remotes for r in remote.recv()]
        obs, self.reset_infos, self._masks = zip(*results)
        self._reset_seeds()
        self._reset_options()
        return np.stack(obs)

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self.processes:
            proc.join(timeout=10)
        self.closed = True

    def _gather(self, indices, make_msg):
        routes = self._route(indices)
        for w, local in routes.items():
            self.remotes[w].send(make_msg(local))
        return [x for w in routes for x in self.remotes[w].recv()]

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        # Masks arrive with every step/reset; any other env_method or set_attr could change
        # them, so those paths clear the cache.
        if method_name == "action_masks" and not method_args and not method_kwargs and self._masks is not None:
            return [self._masks[i] for i in self._get_indices(indices)]
        self._masks = None
        return self._gather(indices, lambda local: ("env_method", (local, method_name, method_args, method_kwargs)))

    def get_attr(self, attr_name, indices=None):
        return self._gather(indices, lambda local: ("get_attr", (local, attr_name)))

    def set_attr(self, attr_name, value, indices=None):
        self._masks = None
        routes = self._route(indices)
        for w, local in routes.items():
            self.remotes[w].send(("set_attr", (local, attr_name, value)))
        for w in routes:
            self.remotes[w].recv()

    def has_attr(self, attr_name):
        self.remotes[0].send(("has_attr", attr_name))
        return self.remotes[0].recv()

    def env_is_wrapped(self, wrapper_class, indices=None):
        # MaskedMonitor provides Monitor's "episode" info; nothing else is wrapped.
        return [False] * len(self._get_indices(indices))

    def get_images(self):
        raise NotImplementedError("rendering is not supported")

