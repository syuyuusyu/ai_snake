from collections import deque

import numpy as np

from game_env import SnakeEnv
from jev_assist import (DIRS, GateState, JevCache, _map_dir, _transforms, detect_gate,
                        head_reaches_tail, safe_mask_fn, safe_moves)


def _env(snake, food, direction):
    env = SnakeEnv(board_size=12, silent_mode=True, seed=0)
    env.game.snake = deque(snake)
    env.game.food = food
    env.game.direction = direction
    return env


def test_safe_mask_blocks_wall_and_body():
    env = _env([(0, 5), (1, 5), (2, 5)], (9, 9), "left")
    assert safe_moves(env.game) == ["up", "down"]
    assert safe_mask_fn(env).tolist() == [1, 1, 0, 0]


def test_safe_mask_falls_back_when_no_move_is_safe():
    env = _env([(0, 0), (1, 0), (1, 1), (0, 1), (0, 2)], (9, 9), "up")
    assert safe_moves(env.game) == []
    assert safe_mask_fn(env).sum() == 3


def test_tail_reachability():
    assert head_reaches_tail([(5, 5), (6, 5), (7, 5)], 12)


def test_loop_gate_fires_on_repeated_state():
    env = _env([(5, 5), (6, 5), (7, 5)], (0, 0), "left")
    gs = GateState()
    assert detect_gate(env.game, "left", gs) is None
    assert detect_gate(env.game, "left", gs) == "loop"


def test_cache_hits_all_symmetries(tmp_path):
    cache = JevCache(tmp_path / "c.jsonl", 12)
    snake, food = [(5, 5), (6, 5), (7, 5), (7, 6)], (2, 3)
    cache.put(snake, food, "up")
    for f in _transforms(12):
        assert cache.get([f(*p) for p in snake], f(*food)) == _map_dir(f, 12, "up")
    reloaded = JevCache(tmp_path / "c.jsonl", 12)
    assert reloaded.get(snake, food) == "up"
