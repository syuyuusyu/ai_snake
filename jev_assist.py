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

DIRS = ["up", "down", "left", "right"]
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}


# --------------------------------------------------------------------------- #
# 1. Free safety checks
# --------------------------------------------------------------------------- #
def safe_moves(game) -> list[str]:
    snake = list(game.snake)
    head = snake[0]
    out = []
    for d in DIRS:
        if d == OPP[game.direction]:
            continue
        t = (head[0] + DELTA[d][0], head[1] + DELTA[d][1])
        if not (0 <= t[0] < game.board_width and 0 <= t[1] < game.board_high):
            continue
        occupied = snake if t == game.food else snake[:-1]
        if t in occupied:
            continue
        out.append(d)
    return out


def game_action_mask(game) -> np.ndarray:
    """Block reverse, wall and body collisions; retain non-reverse moves if trapped."""
    safe = safe_moves(game)
    if not safe:
        return np.array([0 if d == OPP[game.direction] else 1 for d in DIRS], dtype=np.int8)
    return np.array([1 if d in safe else 0 for d in DIRS], dtype=np.int8)


def safe_mask_fn(env) -> np.ndarray:
    return game_action_mask(env.unwrapped.game)


def snake_after(game, d) -> list[tuple[int, int]]:
    snake = list(game.snake)
    h = snake[0]
    t = (h[0] + DELTA[d][0], h[1] + DELTA[d][1])
    return [t] + (snake if t == game.food else snake[:-1])


def head_reaches_tail(snake, board_width, board_high) -> bool:
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
            if 0 <= p[0] < board_width and 0 <= p[1] < board_high and p not in blocked and p not in seen:
                seen.add(p)
                q.append(p)
    return False


def safe_food_move(game) -> str | None:
    """First move of a shortest path to food, if the snake can still reach its tail after eating."""
    snake = list(game.snake)
    head, blocked = snake[0], set(snake[:-1])
    prev, q = {head: None}, deque([head])
    while q and game.food not in prev:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (x + dx, y + dy)
            if 0 <= p[0] < game.board_width and 0 <= p[1] < game.board_high and p not in blocked and p not in prev:
                prev[p] = (x, y)
                q.append(p)
    if game.food not in prev:
        return None
    path, p = [], game.food
    while p != head:
        path.append(p)
        p = prev[p]
    virtual = (path + snake)[:len(snake) + 1]
    if len(virtual) < game.board_width * game.board_high and not head_reaches_tail(
            virtual, game.board_width, game.board_high):
        return None
    step = (path[-1][0] - head[0], path[-1][1] - head[1])
    return next(d for d, v in DELTA.items() if v == step)


class Shield:
    """Narrow the safe moves: keep head->tail reachability, and do not re-enter a
    (snake, food) state seen since the last meal. Each rule yields when nothing passes it.
    After `hunger` steps without food, take a verified-safe shortest path to the food."""

    @staticmethod
    def default_hunger(game) -> int:
        return 2 * game.board_width * game.board_high

    def __init__(self, hunger: int | None = None):
        self.visited = set()
        self.hunger, self.since_food = hunger, 0

    def reset(self):
        self.visited.clear()
        self.since_food = 0

    def record(self, game, ate: bool):
        if ate:
            self.visited.clear()
            self.since_food = 0
        else:
            self.since_food += 1
        self.visited.add((tuple(game.snake), game.food))

    def mask(self, game) -> np.ndarray:
        safe = safe_moves(game)
        if not safe:
            return game_action_mask(game)
        if self.hunger is not None and self.since_food >= self.hunger:
            move = safe_food_move(game)
            if move in safe:
                return np.array([1 if d == move else 0 for d in DIRS], dtype=np.int8)
        after = {d: snake_after(game, d) for d in safe}
        keep = [d for d in safe if head_reaches_tail(after[d], game.board_width, game.board_high)] or safe
        fresh = [d for d in keep if (tuple(after[d]), game.food) not in self.visited] or keep
        return np.array([1 if d in fresh else 0 for d in DIRS], dtype=np.int8)


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
    ok = {d: head_reaches_tail(snake_after(game, d), game.board_width, game.board_high) for d in safe}
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
    step_count: int = 0
    rng_state: object = None

    @staticmethod
    def take(game, ppo_move, gate, obs, mask, step_count=0):
        return Snapshot(list(game.snake), tuple(game.food), game.direction, ppo_move, gate,
                        np.asarray(obs, dtype=np.uint8).copy(), np.asarray(mask, dtype=np.int8).copy(),
                        step_count, game.rng.getstate())


# --------------------------------------------------------------------------- #
# 3a. D4-symmetry canonical cache
# --------------------------------------------------------------------------- #
def _transforms(board_width, board_high):
    w, h = board_width - 1, board_high - 1
    if board_width == board_high:
        return [
            lambda x, y: (x, y), lambda x, y: (w - y, x), lambda x, y: (w - x, h - y), lambda x, y: (y, h - x),
            lambda x, y: (w - x, y), lambda x, y: (x, h - y), lambda x, y: (y, x), lambda x, y: (w - y, h - x),
        ]
    # Only these four symmetries preserve a non-square board's dimensions.
    return [lambda x, y: (x, y), lambda x, y: (w - x, h - y),
            lambda x, y: (w - x, y), lambda x, y: (x, h - y)]


def _map_dir(f, d):
    o = f(0, 0)
    t = f(*DELTA[d])
    v = (t[0] - o[0], t[1] - o[1])
    return next(k for k, dv in DELTA.items() if dv == v)


def canonical(snake, food, board_width, board_high, direction=None, step_count=0):
    """Return the smallest symmetric state, including dimensions in its key."""
    best = None
    for i, f in enumerate(_transforms(board_width, board_high)):
        key = (board_width, board_high, tuple(f(*p) for p in snake), f(*food),
               _map_dir(f, direction) if direction is not None else "", step_count)
        if best is None or key < best[0]:
            best = (key, i)
    return best


class JevCache:
    """Cache dimension-preserving symmetries with label provenance and weight.

    Legacy records without provenance are ignored: they cannot safely be used
    in the rollout-only control or assigned a verified-label weight.
    """

    def __init__(self, path: str | Path, board_width: int, board_high: int):
        self.path, self.lock = Path(path), threading.Lock()
        self.board_width, self.board_high = board_width, board_high
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {}
        if self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("source") in ("rollout", "jev"):
                        self.data[rec["k"]] = (rec["a"], rec["source"])

    def _k(self, snake, food, direction=None, step_count=0):
        key, ti = canonical(snake, food, self.board_width, self.board_high, direction, step_count)
        return json.dumps(key, separators=(",", ":")), ti

    def get_label(self, snake, food, direction=None, step_count=0, allow_jev=True):
        k, ti = self._k(snake, food, direction, step_count)
        entry = self.data.get(k)
        if entry is None:
            return None
        action, source = entry
        if source == "jev" and not allow_jev:
            return None
        f = _transforms(self.board_width, self.board_high)[ti]
        move = next(d for d in DIRS if _map_dir(f, d) == action)
        return move, 1.0 if source == "rollout" else 0.5, source

    def get(self, snake, food, direction=None, step_count=0):
        label = self.get_label(snake, food, direction, step_count)
        return label[0] if label else None

    def put(self, snake, food, move, source="rollout", direction=None, step_count=0):
        k, ti = self._k(snake, food, direction, step_count)
        action = _map_dir(_transforms(self.board_width, self.board_high)[ti], move)
        with self.lock:
            old = self.data.get(k)
            # A verified rollout can replace a weaker Jev-only label.
            if old is None or (old[1] == "jev" and source == "rollout"):
                self.data[k] = (action, source)
                with self.path.open("a") as fh:
                    fh.write(json.dumps({"k": k, "a": action, "source": source}) + "\n")


# --------------------------------------------------------------------------- #
# 3b/3c. Labeler: rollouts first, Jev only if inconclusive
# --------------------------------------------------------------------------- #
class Labeler:
    def __init__(self, make_env, policy, jev_fn, cache: JevCache, horizon=200,
                 audit_rate=0.05, max_workers=4, budget_per_round=32, score_margin=50.0, seed=0):
        """
        make_env : () -> SnakeEnv used only for rollouts (main process)
        policy   : MaskableActorCriticPolicy on CPU (a frozen copy is fine)
        jev_fn   : (game) -> direction | None   (wraps build_state/build_request/ask_jev)
        """
        self.env = make_env()
        self.policy, self.jev_fn, self.cache = policy, jev_fn, cache
        self.horizon, self.audit_rate, self.budget = horizon, audit_rate, budget_per_round
        self.score_margin = score_margin
        self.rng = random.Random(seed)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.stats = dict(gates=0, cache_hit=0, decisive=0, jev_calls=0, jev_fail=0,
                          audit=0, audit_agree=0, skipped_budget=0)

    # -- rollouts --------------------------------------------------------- #
    def _load(self, snap: Snapshot):
        e = copy.deepcopy(self.env)
        g = e.game
        g.snake = deque(snap.snake)
        g.food, g.direction = snap.food, snap.direction
        e.step_count = snap.step_count
        if snap.rng_state is not None:
            g.rng.setstate(snap.rng_state)
        e._safety_score = g.safety_score()
        return e

    def _greedy(self, e):
        import torch

        obs = e.game.get_obs()
        t, _ = self.policy.obs_to_tensor(obs)
        with torch.no_grad():
            p = self.policy.get_distribution(t, action_masks=safe_mask_fn(e)).distribution.probs[0]
        return int(torch.argmax(p))

    def _rollout(self, snap, first):
        e = self._load(snap)
        _, _, term, trunc, info = e.step(DIRS.index(first))
        eaten, steps = int(info["step_state"] == "eat food"), 1
        while not (term or trunc) and steps < self.horizon:
            _, _, term, trunc, info = e.step(self._greedy(e))
            steps += 1
            eaten += info["step_state"] == "eat food"
        if info["step_state"] == "Victory":
            return 1e6, False
        died = bool(term or trunc)
        return steps + 50 * eaten - (1e4 if died else 0), died

    def verify(self, snap):
        """Compare branches from identical private RNG and timeout states."""
        res = {d: self._rollout(snap, d) for d in safe_moves(self._load(snap).game)}
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
            hit = self.cache.get_label(s.snake, s.food, s.direction, s.step_count,
                                       allow_jev=self.jev_fn is not None)
            if hit is not None and s.mask[DIRS.index(hit[0])]:
                self.stats["cache_hit"] += 1
                move, weight, source = hit
                out.append((s.obs, s.mask, DIRS.index(move), weight, source))
                continue
            best, decisive, _ = self.verify(s)
            if decisive:
                self.stats["decisive"] += 1
                out.append((s.obs, s.mask, DIRS.index(best), 1.0, "rollout"))
                self.cache.put(s.snake, s.food, best, "rollout", s.direction, s.step_count)
                if self.jev_fn is not None and self.rng.random() < self.audit_rate and len(jev_jobs) < self.budget:
                    jev_jobs.append((s, best, True))
                continue
            if self.jev_fn is None or len(jev_jobs) >= self.budget:
                self.stats["skipped_budget"] += 1
                continue
            jev_jobs.append((s, best, False))

        self.stats["jev_calls"] += len(jev_jobs)
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
            self.cache.put(s.snake, s.food, move, "jev", s.direction, s.step_count)
            # Jev alone is weaker than PPO; weight its labels below verified ones
            out.append((s.obs, s.mask, DIRS.index(move), 0.5, "jev"))
        return out

    def _ask(self, snap):
        try:
            game = self._load(snap).game
            move = self.jev_fn(game)
            return move if move in safe_moves(game) and snap.mask[DIRS.index(move)] else None
        except Exception as exc:  # network errors must never kill training
            print(f"[jev] call failed: {exc}")
            return None

    def close(self):
        self.pool.shutdown(wait=True)
        self.env.close()


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
        import torch

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
        move, _ = ask_jev(build_request(state, model_id), max_attempts=1)
        return move
    return fn
