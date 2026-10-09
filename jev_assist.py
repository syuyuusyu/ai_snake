"""Jev-assisted training helpers: gating, caching, verification and label storage.

Design (cheap checks first, Jev last):
  1. safe_moves / safe_mask_fn  : mask immediately lethal moves (free).
  2. detect_gate                : flag only "critical" states (free BFS checks):
       - risk : PPO's move cuts head->tail reachability while another safe move keeps it
       - loop : the exact (snake, food) state repeats without eating
  3. Labeler.label              : for each flagged state
       a. canonical (D4 symmetry) cache lookup              -> 0 calls on hit
       b. short policy rollouts from every safe move        -> 0 calls if decisive
          (a branch dies while another survives, or the score gap >= one food)
       c. ask Jev only when rollouts are inconclusive       -> 1 call
       d. audit: with small probability also ask Jev on decisive states to
          measure Jev's agreement with the rollout verdict
  4. LabelBuffer                : (obs, mask, action, weight) used by an auxiliary
                                  behaviour-cloning loss; PPO itself stays on-policy.
"""

from __future__ import annotations

import copy
import json
import random
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

DIRS = ["up", "down", "left", "right"]
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}


# --------------------------------------------------------------------------- #
# 1. Free safety checks
# --------------------------------------------------------------------------- #
def safe_moves(game) -> list[str]:
    snake = list(game.snake)
    head, n = snake[0], game.board_size
    out = []
    for d in DIRS:
        if d == OPP[game.direction]:
            continue
        t = (head[0] + DELTA[d][0], head[1] + DELTA[d][1])
        if not (0 <= t[0] < n and 0 <= t[1] < n):
            continue
        occupied = snake if t == game.food else snake[:-1]
        if t in occupied:
            continue
        out.append(d)
    return out


def safe_mask_fn(env) -> np.ndarray:
    """ActionMasker callback: mask reverse AND immediately lethal moves.
    Falls back to the non-reverse mask when no move is safe (death is forced)."""
    game = env.unwrapped.game
    safe = safe_moves(game)
    if not safe:
        return np.array([0 if d == OPP[game.direction] else 1 for d in DIRS], dtype=np.int8)
    return np.array([1 if d in safe else 0 for d in DIRS], dtype=np.int8)


def snake_after(game, d) -> list[tuple[int, int]]:
    snake = list(game.snake)
    h = snake[0]
    t = (h[0] + DELTA[d][0], h[1] + DELTA[d][1])
    return [t] + (snake if t == game.food else snake[:-1])


def head_reaches_tail(snake, n) -> bool:
    if len(snake) <= 2:
        return True
    head, tail = snake[0], snake[-1]
    blocked = set(snake[1:-1])
    q, seen = deque([head]), {head}
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (x + dx, y + dy)
            if p == tail:
                return True
            if 0 <= p[0] < n and 0 <= p[1] < n and p not in blocked and p not in seen:
                seen.add(p)
                q.append(p)
    return False


# --------------------------------------------------------------------------- #
# 2. Gate: decide whether a state is worth a label at all
# --------------------------------------------------------------------------- #
@dataclass
class GateState:
    seen: dict = field(default_factory=dict)

    def on_eat(self):
        self.seen.clear()


def detect_gate(game, ppo_move: str, gate_state: GateState) -> str | None:
    safe = safe_moves(game)
    if len(safe) < 2:
        return None
    key = (tuple(game.snake), game.food)
    looped = key in gate_state.seen
    gate_state.seen[key] = True
    n = game.board_size
    ok = {d: head_reaches_tail(snake_after(game, d), n) for d in safe}
    if ppo_move in ok and not ok[ppo_move] and any(ok.values()):
        return "risk"
    if looped:
        return "loop"
    return None


@dataclass
class Snapshot:
    snake: list
    food: tuple
    direction: str
    ppo_move: str
    gate: str
    obs: np.ndarray
    mask: np.ndarray

    @staticmethod
    def take(game, ppo_move, gate, obs, mask):
        return Snapshot(list(game.snake), tuple(game.food), game.direction, ppo_move, gate,
                        np.asarray(obs, dtype=np.uint8).copy(), np.asarray(mask, dtype=np.int8).copy())


# --------------------------------------------------------------------------- #
# 3a. D4-symmetry canonical cache
# --------------------------------------------------------------------------- #
def _transforms(n):
    m = n - 1
    return [
        lambda x, y: (x, y), lambda x, y: (m - y, x), lambda x, y: (m - x, m - y), lambda x, y: (y, m - x),
        lambda x, y: (m - x, y), lambda x, y: (x, m - y), lambda x, y: (y, x), lambda x, y: (m - y, m - x),
    ]


def _map_dir(f, n, d):
    o = f(n // 2, n // 2)
    t = f(n // 2 + DELTA[d][0], n // 2 + DELTA[d][1])
    v = (t[0] - o[0], t[1] - o[1])
    return next(k for k, dv in DELTA.items() if dv == v)


def canonical(snake, food, n):
    """Return (key, transform_index) where key is the lexicographically smallest
    image of the state under the 8 board symmetries."""
    best = None
    for i, f in enumerate(_transforms(n)):
        key = (tuple(f(*p) for p in snake), f(*food))
        if best is None or key < best[0]:
            best = (key, i)
    return best


class JevCache:
    """Persistent answer cache keyed by canonical state. Stores the answer in the
    canonical frame so any of the 8 symmetric states hits the same entry."""

    def __init__(self, path: str | Path, n: int):
        self.path, self.n, self.lock = Path(path), n, threading.Lock()
        self.data: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.open():
                rec = json.loads(line)
                self.data[rec["k"]] = rec["a"]

    def _k(self, snake, food):
        key, ti = canonical(snake, food, self.n)
        return json.dumps(key, separators=(",", ":")), ti

    def get(self, snake, food):
        k, ti = self._k(snake, food)
        a = self.data.get(k)
        if a is None:
            return None
        f = _transforms(self.n)[ti]
        # invert: find original direction whose image is the cached canonical direction
        return next(d for d in DIRS if _map_dir(f, self.n, d) == a)

    def put(self, snake, food, move):
        k, ti = self._k(snake, food)
        a = _map_dir(_transforms(self.n)[ti], self.n, move)
        with self.lock:
            if k not in self.data:
                self.data[k] = a
                with self.path.open("a") as fh:
                    fh.write(json.dumps({"k": k, "a": a}) + "\n")


# --------------------------------------------------------------------------- #
# 3b/3c. Labeler: rollouts first, Jev only if inconclusive
# --------------------------------------------------------------------------- #
class Labeler:
    def __init__(self, make_env, policy, jev_fn, cache: JevCache, horizon=200,
                 audit_rate=0.05, max_workers=4, budget_per_round=64, score_margin=50.0):
        """
        make_env : () -> SnakeEnv used only for rollouts (main process)
        policy   : MaskableActorCriticPolicy on CPU (a frozen copy is fine)
        jev_fn   : (game) -> direction | None   (wraps build_state/build_request/ask_jev)
        """
        self.env = make_env()
        self.policy, self.jev_fn, self.cache = policy, jev_fn, cache
        self.horizon, self.audit_rate, self.budget = horizon, audit_rate, budget_per_round
        self.score_margin = score_margin
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.stats = dict(gates=0, cache_hit=0, decisive=0, jev_calls=0, jev_fail=0,
                          audit=0, audit_agree=0, skipped_budget=0)

    # -- rollouts --------------------------------------------------------- #
    def _load(self, snap: Snapshot):
        e = self.env
        g = e.game
        g.snake = deque(snap.snake)
        g.food, g.direction = snap.food, snap.direction
        e.step_count = 0
        e._safety_score = g.safety_score()
        return e

    def _greedy(self, e):
        obs = e.game.get_obs()
        t, _ = self.policy.obs_to_tensor(obs)
        with torch.no_grad():
            p = self.policy.get_distribution(t, action_masks=safe_mask_fn(e)).distribution.probs[0]
        return int(torch.argmax(p))

    def _rollout(self, snap, first):
        e = copy.deepcopy(self._load(snap))
        _, _, term, trunc, info = e.step(DIRS.index(first))
        eaten, steps = int(info["step_state"] == "eat food"), 1
        while not (term or trunc) and steps < self.horizon:
            _, _, term, trunc, info = e.step(self._greedy(e))
            steps += 1
            eaten += info["step_state"] == "eat food"
        if info["step_state"] == "Victory":
            return 1e6, False
        died = bool(term)
        return steps + 50 * eaten - (1e4 if died else 0), died

    def verify(self, snap):
        """Returns (best_move, decisive). Restores the global RNG used for food."""
        rng = random.getstate()
        res = {}
        for d in safe_moves(self._load(snap).game):
            random.setstate(rng)
            res[d] = self._rollout(snap, d)
        random.setstate(rng)
        died = [v[1] for v in res.values()]
        ranked = sorted((v[0] for v in res.values()), reverse=True)
        gap = ranked[0] - ranked[1] if len(ranked) > 1 else float("inf")
        # decisive: some branches die and others survive, or the best branch wins
        # by at least `score_margin` (default: one extra food within the horizon)
        decisive = (any(died) and not all(died)) or gap >= self.score_margin
        best = max(res, key=lambda d: res[d][0])
        return best, decisive, res

    # -- main entry ------------------------------------------------------- #
    def label(self, snaps: list[Snapshot]):
        """Returns list of (obs, mask, action_idx, weight, source)."""
        out, jev_jobs = [], []
        for s in snaps:
            self.stats["gates"] += 1
            hit = self.cache.get(s.snake, s.food)
            if hit is not None:
                self.stats["cache_hit"] += 1
                out.append((s.obs, s.mask, DIRS.index(hit), 1.0, "cache"))
                continue
            best, decisive, _ = self.verify(s)
            if decisive:
                self.stats["decisive"] += 1
                out.append((s.obs, s.mask, DIRS.index(best), 1.0, "rollout"))
                self.cache.put(s.snake, s.food, best)
                if random.random() < self.audit_rate and len(jev_jobs) < self.budget:
                    jev_jobs.append((s, best, True))
                continue
            if len(jev_jobs) >= self.budget:
                self.stats["skipped_budget"] += 1
                continue
            jev_jobs.append((s, best, False))

        futures = [(self.pool.submit(self._ask, s), s, best, audit) for s, best, audit in jev_jobs]
        for fut, s, best, audit in futures:
            move = fut.result()
            if move is None:
                self.stats["jev_fail"] += 1
                continue
            if audit:
                self.stats["audit"] += 1
                self.stats["audit_agree"] += int(move == best)
                continue  # rollout label already stored
            self.cache.put(s.snake, s.food, move)
            # Jev alone is weaker than PPO; weight its labels below verified ones
            out.append((s.obs, s.mask, DIRS.index(move), 0.5, "jev"))
        return out

    def _ask(self, snap):
        self.stats["jev_calls"] += 1
        try:
            return self.jev_fn(self._load(copy.deepcopy(snap)).game)
        except Exception as exc:  # network errors must never kill training
            print(f"[jev] call failed: {exc}")
            return None


# --------------------------------------------------------------------------- #
# 4. Label buffer for the auxiliary BC loss
# --------------------------------------------------------------------------- #
class LabelBuffer:
    def __init__(self, capacity=50_000):
        self.items = deque(maxlen=capacity)

    def add(self, items):
        self.items.extend(items)

    def __len__(self):
        return len(self.items)

    def sample(self, batch_size, device):
        idx = np.random.randint(len(self.items), size=batch_size)
        obs, mask, act, w = zip(*[self.items[i][:4] for i in idx])
        return (torch.as_tensor(np.stack(obs), device=device).float(),
                torch.as_tensor(np.stack(mask), device=device),
                torch.as_tensor(act, device=device).long(),
                torch.as_tensor(w, device=device).float())


def make_jev_fn(model_id="typesafe/jev-1.13"):
    """Adapter around the existing play_with_jev helpers."""
    from play_with_jev import ask_jev, build_request, build_state

    def fn(game):
        state = build_state(game)
        if len(state["safe_moves"]) < 2:
            return next(iter(state["safe_moves"]), None)
        move, _ = ask_jev(build_request(state, model_id))
        return move
    return fn
