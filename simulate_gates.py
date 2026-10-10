"""Usage: python simulate_gates.py {safe|tail|oracle} N_SEEDS OUT.jsonl [START_SEED]

Simulate gating strategies for an external advisor on top of stable_4.

modes:
  safe     : safety mask only
  tail     : safety mask + rule: if PPO move loses head->tail reachability and an alternative keeps it, switch (0 advisor calls)
  oracle   : safety mask; at gate points, ask an "oracle" advisor = best move by short PPO rollouts (upper bound for any advisor)
Gate points (where an advisor would be called):
  G1 risk  : n_safe>1 and the PPO move breaks tail reachability (or lands in smaller region) while an alternative does not
  G2 loop  : the exact (snake, food) state repeats -> policy is cycling
"""
import argparse
import copy
import json
import random
from collections import deque

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from board_config import add_board_arguments
from game_env import SnakeEnv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["safe", "tail", "oracle"])
parser.add_argument("seeds", type=int)
parser.add_argument("out")
parser.add_argument("start_seed", type=int, nargs="?", default=0)
add_board_arguments(parser)
args = parser.parse_args()
MODE, N, OUT = args.mode, args.seeds, args.out
SEEDS = range(args.start_seed, args.start_seed + N)
DIRS = ['up', 'down', 'left', 'right']; D = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
OPP = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
env0 = SnakeEnv(board_width=args.board_width, board_high=args.board_high, silent_mode=True, seed=0, bfs_intensity=1)
model = MaskablePPO.load("pth/stable_4.zip", device="cpu", custom_objects={
    "observation_space": env0.observation_space, "action_space": env0.action_space,
    "learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0})
pol = model.policy; pol.set_training_mode(False)


def safe_moves(g):
    snake = list(g.snake); h = snake[0]; out = []
    for d in DIRS:
        if d == OPP[g.direction]: continue
        t = (h[0] + D[d][0], h[1] + D[d][1])
        if not (0 <= t[0] < g.board_width and 0 <= t[1] < g.board_high): continue
        occ = snake if t == g.food else snake[:-1]
        if t in occ: continue
        out.append(d)
    return out


def after(g, d):
    """snake body after moving d (no food respawn, no RNG use)."""
    snake = list(g.snake); h = snake[0]; t = (h[0] + D[d][0], h[1] + D[d][1])
    return [t] + (snake if t == g.food else snake[:-1])


def tail_ok(snake, board_width, board_high):
    """head can reach tail through free cells (tail cell counts as free)."""
    h, tail = snake[0], snake[-1]; blocked = set(snake[1:-1])
    q = deque([h]); seen = {h}
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (x + dx, y + dy)
            if p == tail and len(snake) > 2: return True
            if 0 <= p[0] < board_width and 0 <= p[1] < board_high and p not in blocked and p not in seen and p != tail:
                seen.add(p); q.append(p)
    return False


def region(snake, board_width, board_high):
    h = snake[0]; blocked = set(snake[1:]); q = deque([h]); seen = {h}
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (x + dx, y + dy)
            if 0 <= p[0] < board_width and 0 <= p[1] < board_high and p not in blocked and p not in seen:
                seen.add(p); q.append(p)
    return len(seen) - 1


def policy_probs(env, mask):
    t, _ = pol.obs_to_tensor(env.game.get_obs())
    with torch.no_grad():
        return pol.get_distribution(t, action_masks=mask).distribution.probs[0].numpy()


def mask_of(g, safe):
    if not safe:
        return np.array([0 if d == OPP[g.direction] else 1 for d in DIRS], dtype=np.int8)
    return np.array([1 if d in safe else 0 for d in DIRS], dtype=np.int8)


def rollout_value(env, first, horizon=300):
    """Play `first` then PPO(safe-masked) greedily; score = survival + food. RNG state restored by caller."""
    e = copy.deepcopy(env)
    _, _, term, trunc, info = e.step(DIRS.index(first))
    eaten = int(info['step_state'] == 'eat food'); steps = 1
    if term: return (1e6 if info['step_state'] == 'Victory' else -1e4)
    while steps < horizon:
        s = safe_moves(e.game)
        if not s: return steps + 50 * eaten - 1e4
        a = int(np.argmax(policy_probs(e, mask_of(e.game, s))))
        _, _, term, trunc, info = e.step(a); steps += 1
        if info['step_state'] == 'eat food': eaten += 1
        if term:
            return 1e6 if info['step_state'] == 'Victory' else steps + 50 * eaten - 1e4
        if trunc: break
    return steps + 50 * eaten


DECISIVE = [0, 0]
def oracle(env, cands):
    st = random.getstate(); best, bv = None, -1e18; vals = []
    for d in cands:
        random.setstate(st)
        v = rollout_value(env, d); vals.append(v)
        if v > bv: best, bv = d, v
    random.setstate(st)
    dies = [v < 0 for v in vals]
    if len(vals) > 1:
        DECISIVE[0 if (any(dies) and not all(dies)) else 1] += 1
    return best


with open(OUT, 'w') as f:
    for seed in SEEDS:
        env = SnakeEnv(board_width=args.board_width, board_high=args.board_high, silent_mode=True, seed=seed, bfs_intensity=1)
        env.reset()
        DECISIVE[0] = DECISIVE[1] = 0
        maxlen = 4; calls = {'risk': 0, 'loop': 0}; overrides = 0; seen_states = {}
        risk_steps = []
        for step in range(1, 20001):
            g = env.game; safe = safe_moves(g)
            p = policy_probs(env, mask_of(g, safe)); move = DIRS[int(np.argmax(p))]
            gate = None
            if len(safe) > 1:
                ok = {d: tail_ok(after(g, d), g.board_width, g.board_high) for d in safe}
                if not ok[move] and any(ok.values()):
                    gate = 'risk'
                key = (tuple(g.snake), g.food)
                if key in seen_states and gate is None:
                    gate = 'loop'
                seen_states[key] = step
            if gate:
                calls[gate] += 1; risk_steps.append((step, len(g.snake), gate))
                if MODE == 'tail' and gate == 'risk':
                    alt = [d for d in safe if ok[d]]
                    new = max(alt, key=lambda d: p[DIRS.index(d)])
                elif MODE == 'oracle':
                    cands = safe if gate == 'loop' else safe
                    new = oracle(env, cands)
                    if gate == 'loop' and new == move:  # break cycle: best non-PPO move
                        others = [d for d in safe if d != move]
                        new = oracle(env, others)
                    seen_states.clear()
                else:
                    new = move
                if new != move: overrides += 1
                move = new
            _, _, term, trunc, info = env.step(DIRS.index(move))
            if info['step_state'] == 'eat food': seen_states.clear()
            maxlen = max(maxlen, info['snake_length'])
            if term or trunc:
                outcome = 'victory' if info['step_state'] == 'Victory' else ('death' if term else 'timeout'); break
        rec = dict(seed=seed, outcome=outcome, steps=step, max_length=maxlen, calls=calls, overrides=overrides, decisive=DECISIVE[0], inconclusive=DECISIVE[1],
                   gate_steps=risk_steps[-30:])
        f.write(json.dumps(rec) + '\n'); f.flush()
        print(seed, outcome, step, maxlen, calls, overrides, DECISIVE, flush=True)
