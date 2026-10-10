from collections import deque

import numpy as np

from game_env import SnakeEnv
from jev_assist import (
    DIRS,
    GateState,
    JevCache,
    Shield,
    _map_dir,
    _transforms,
    detect_gate,
    head_reaches_tail,
    safe_food_move,
    safe_mask_fn,
    safe_moves,
)


def _env(snake, food, direction):
    env = SnakeEnv(board_width=12, board_high=12, silent_mode=True, seed=0)
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
    assert head_reaches_tail([(5, 5), (6, 5), (7, 5)], 12, 12)


def test_shield_blocks_move_that_cuts_off_tail():
    # A body wall along x=3 splits the board; the tail lies on the right side.
    snake = [(3, y) for y in range(12)] + [(4, 11), (5, 11)]
    env = _env(snake, (9, 5), "up")
    assert safe_moves(env.game) == ["left", "right"]
    assert Shield().mask(env.game).tolist() == [0, 0, 0, 1]


def test_shield_avoids_revisiting_until_food_is_eaten():
    env = _env([(5, 5), (6, 5), (7, 5)], (0, 0), "left")
    shield = Shield()
    shield.record(_env([(4, 5), (5, 5), (6, 5)], (0, 0), "left").game, ate=False)
    assert shield.mask(env.game).tolist() == [1, 1, 0, 0]
    shield.record(_env([(9, 9), (9, 10), (9, 11)], (0, 0), "up").game, ate=True)
    assert shield.mask(env.game).tolist() == [1, 1, 1, 0]


def test_hungry_shield_takes_safe_path_to_food():
    env = _env([(5, 5), (6, 5), (7, 5)], (5, 2), "left")
    assert safe_food_move(env.game) == "up"
    shield = Shield(hunger=1)
    assert shield.mask(env.game).tolist() == [1, 1, 1, 0]
    shield.record(env.game, ate=False)
    assert shield.mask(env.game).tolist() == [1, 0, 0, 0]


def test_safe_food_move_rejects_food_inside_a_pocket():
    # Food sits in a dead-end pocket; eating it would leave the head sealed off from the tail.
    snake = [(1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]
    env = _env(snake, (2, 2), "down")
    assert safe_food_move(env.game) is None


def test_loop_gate_fires_on_repeated_state():
    env = _env([(5, 5), (6, 5), (7, 5)], (0, 0), "left")
    gs = GateState()
    assert detect_gate(env.game, "left", gs) is None
    assert detect_gate(env.game, "left", gs) == "loop"


def test_cache_hits_all_symmetries(tmp_path):
    cache = JevCache(tmp_path / "c.jsonl", 12, 12)
    snake, food = [(5, 5), (6, 5), (7, 5), (7, 6)], (2, 3)
    cache.put(snake, food, "up")
    for f in _transforms(12, 12):
        assert cache.get([f(*p) for p in snake], f(*food)) == _map_dir(f, "up")
    reloaded = JevCache(tmp_path / "c.jsonl", 12, 12)
    assert reloaded.get(snake, food) == "up"
