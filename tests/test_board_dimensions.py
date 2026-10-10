import argparse
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from board_config import add_board_arguments
from game_env import SnakeEnv
from jev_assist import JevCache, _map_dir, _transforms, head_reaches_tail, safe_moves
from module import SnakeNet
from play_with_jev import build_state
from play_with_model import make_env as make_play_env
from snake_game import SnakeGame
from train_with_jev import SafeFixedSeedEval
from train_with_jev import make_env as make_jev_env
from train_with_ppo import FixedSeedEvalCallback
from train_with_ppo import make_env as make_ppo_env


class BoardDimensionsTest(unittest.TestCase):
    def test_default_board_and_observation_remain_compatible(self):
        parser = argparse.ArgumentParser()
        add_board_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual((args.board_width, args.board_high), (12, 12))
        for factory in (SnakeEnv, SnakeGame):
            obj = factory()
            game = obj.game if isinstance(obj, SnakeEnv) else obj
            self.assertEqual((game.board_width, game.board_high), (12, 12))
            self.assertEqual(game.get_obs().shape, (3, 42, 42))
            self.assertEqual(list(game.snake), [(6, 6), (7, 6), (8, 6), (9, 6)])
            obj.close()

    def test_dimensions_pass_through_all_environment_factories(self):
        parser = argparse.ArgumentParser()
        add_board_arguments(parser)
        args = parser.parse_args(["--board-width", "8", "--board-high", "13"])
        for factory in (make_ppo_env, make_jev_env, make_play_env):
            with self.subTest(factory=factory):
                env = factory(7, args.board_width, args.board_high)()
                try:
                    obs, _ = env.reset()
                    game = env.unwrapped.game
                    self.assertEqual((game.board_width, game.board_high), (8, 13))
                    self.assertEqual(obs.shape, (3, 40, 60))
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertEqual(env.unwrapped.max_snake_length, 104)
                    self.assertEqual(env.unwrapped.repeat_prossibility.shape, (8, 13))
                    env.unwrapped.reset_rollout()
                    env.step(0)
                    self.assertEqual(env.unwrapped.repeat_prossibility.shape, (8, 13))
                finally:
                    env.close()

    def test_rectangular_food_bfs_and_observation_coordinates(self):
        for width, high in ((8, 13), (13, 8), (4, 2)):
            with self.subTest(width=width, high=high):
                env = SnakeEnv(width, high)
                game = env.game
                for _ in range(5):
                    obs, _ = env.reset()
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertTrue(all(0 <= x < width and 0 <= y < high for x, y in game.snake))
                    self.assertNotIn(game.food, game.snake)
                    self.assertTrue(0 <= game.food[0] < width and 0 <= game.food[1] < high)
                self.assertEqual(len(game.bfs_reachable_area()), width * high - 3)
                self.assertAlmostEqual(game.safety_score(), 1.0)
                game.food = (width - 1, high - 1)
                if game.food in game.snake:
                    game.food = (width - 1, 0)
                food_x, food_y = game.food
                np.testing.assert_array_equal(game.get_play_ground()[food_x, food_y], game.green)
                state = build_state(game)
                self.assertEqual((state["board_width"], state["board_high"]), (width, high))
                self.assertEqual((len(state["board_rows"]), len(state["board_rows"][0])), (high, width))
                self.assertEqual(state["board_rows"][food_y][food_x], "F")
                self.assertEqual(state["free_cells"], width * high - 4)
                env.close()

    def test_dimensions_survive_subprocess_serialization(self):
        env = SubprocVecEnv([make_ppo_env(0, 8, 13), make_jev_env(1, 8, 13)], start_method="spawn")
        try:
            self.assertEqual(env.reset().shape, (2, 3, 40, 60))
            self.assertEqual(env.get_attr("max_snake_length"), [104, 104])
            self.assertEqual(env.step([0, 0])[0].shape, (2, 3, 40, 60))
        finally:
            env.close()

    def test_timeout_uses_rectangle_area(self):
        env = SnakeEnv(8, 13)
        env.game.food = (0, 0)
        env.step_count = 8 * 13 * 4 - 1
        _, _, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reward_timeout"], -1.0)
        env.close()

    def test_walls_safe_moves_and_edges_use_correct_axis(self):
        cases = [(7, 10, "right"), (4, 12, "down")]
        for x, y, direction in cases:
            env = SnakeEnv(8, 13)
            env.game.snake = deque([(x, y), (x - 1, y), (x - 2, y)])
            env.game.food = (0, 0)
            env.game.direction = direction
            self.assertNotIn(direction, safe_moves(env.game))
            self.assertTrue(env.is_on_edge((x, y)))
            self.assertTrue(env.is_on_right_and_down((x, y)))
            self.assertEqual(env.is_on_right((x, y)), x == 7)
            _, _, terminated, _, info = env.step(env.game.directions.index(direction))
            self.assertTrue(terminated)
            self.assertEqual(info["step_state"], "hit wall")
            env.close()

    def test_rectangle_tail_path_cannot_escape_through_short_axis(self):
        # Body divides the entire width; a square-width/high mixup creates an escape.
        snake = [(0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (3, 3)]
        self.assertFalse(head_reaches_tail(snake, 4, 8))
        self.assertTrue(head_reaches_tail(snake, 8, 8))

    def test_victory_uses_rectangle_area(self):
        env = SnakeEnv(4, 2)
        env.game.snake = deque([(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (2, 1), (1, 1)])
        env.game.food = (0, 1)
        env.game.direction = "left"
        self.assertTrue(build_state(env.game)["safe_moves"]["down"]["wins"])
        _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["step_state"], "Victory")
        self.assertEqual(info["snake_length"], 8)
        env.close()

    def test_rectangular_cache_symmetries_and_dimension_isolation(self):
        snake, food = [(2, 3), (3, 3), (4, 3), (4, 4)], (1, 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            cache = JevCache(path, 8, 13)
            cache.put(snake, food, "up")
            self.assertEqual(len(_transforms(8, 13)), 4)
            for transform in _transforms(8, 13):
                self.assertEqual(cache.get([transform(*p) for p in snake], transform(*food)),
                                 _map_dir(transform, "up"))
            self.assertEqual(JevCache(path, 8, 13).get(snake, food), "up")
            self.assertIsNone(JevCache(path, 13, 8).get(snake, food))
            self.assertIsNone(JevCache(path, 12, 12).get(snake, food))

    def test_eval_callbacks_use_requested_dimensions(self):
        for callback_class in (FixedSeedEvalCallback, SafeFixedSeedEval):
            callback = callback_class(1, [0], "unused", board_width=8, board_high=13)
            callback.model = Mock(policy_kwargs={})
            with patch.object(callback, "_evaluate_and_save"):
                callback._on_training_start()
            self.assertEqual(callback.eval_env.observation_space.shape, (3, 40, 60))
            self.assertEqual(callback.eval_env.max_snake_length, 104)
            with patch.object(callback, "_evaluate_and_save"):
                callback._on_training_end()

    def test_eval_victory_score_uses_rectangle_area(self):
        callback = FixedSeedEvalCallback(1, [0], "unused", verbose=0, board_width=8, board_high=13)
        callback.eval_env = SnakeEnv(8, 13)
        callback.eval_policy = Mock()
        callback.eval_policy.predict.return_value = (0, None)
        callback.model = Mock()
        callback.model.policy.state_dict.return_value = {}
        with patch.object(callback.eval_env, "step", return_value=(None, 30, True, False,
                          {"snake_length": 104, "step_state": "Victory"})):
            callback._evaluate_and_save()
        callback.model.logger.record.assert_any_call("eval/score", 208.0)
        callback._on_training_end()

    def test_display_dimensions_and_drawing(self):
        with patch("snake_game.pygame") as pygame:
            game = SnakeGame(8, 13, silent_mode=False)
            pygame.display.set_mode.assert_called_once_with((800, 1300))
            game.draw()
            self.assertEqual(pygame.draw.rect.call_count, 104)
            game.draw_obs()
            game.close()

    def test_rectangular_cnn_can_train_and_predict(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        env = make_jev_env(0, 8, 13)()
        try:
            model = MaskablePPO("CnnPolicy", env, n_steps=2, batch_size=2, n_epochs=1, device="cpu")
            model.learn(total_timesteps=2)
            obs, _ = env.reset()
            action, _ = model.predict(obs, action_masks=env.unwrapped.mask_fn(), deterministic=True)
            self.assertIn(int(action), range(4))
            self.assertEqual(SnakeNet(8, 13)(torch.zeros(1, 1, 13, 8)).shape, (1, 4))
        finally:
            env.close()
            torch.set_num_threads(previous_threads)

    def test_invalid_dimensions_rejected(self):
        for width, high in ((0, 12), (12, 0), (3, 12), (12, -1), (8.5, 12)):
            with self.subTest(width=width, high=high), self.assertRaises(ValueError):
                SnakeEnv(width, high)
