import unittest
from collections import deque
from unittest.mock import patch

import numpy as np

from game_env import SnakeEnv


class SnakeEnvStepTest(unittest.TestCase):
    def test_victory_receives_victory_reward(self):
        env = SnakeEnv(board_size=4)
        observation = np.zeros(env.observation_space.shape, dtype=np.uint8)

        with (
            patch.object(env.game, "step", return_value=(True, 5)),
            patch.object(env, "_get_obs", return_value=observation),
            patch.object(env.game, "safety_score", return_value=0.8),
        ):
            result = env.step(0)

        _, reward, terminated, truncated, info = result

        self.assertGreater(reward, 9.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["step_state"], "Victory")
        self.assertEqual(info["reward_victory"], 10.0)
        self.assertEqual(info["task_reward"], 10.0)
        self.assertEqual(info["reward_total"], reward)
        self.assertEqual(env.victory_count, 1)

    def test_death_penalty_grows_with_snake_length(self):
        short_env = SnakeEnv(board_size=12)
        long_env = SnakeEnv(board_size=12)
        long_env.game.snake = deque((index, 0) for index in range(80))

        def take_terminal_step(env):
            observation = np.zeros(env.observation_space.shape, dtype=np.uint8)
            with (
                patch.object(env.game, "step", return_value=(True, 3)),
                patch.object(env, "_get_obs", return_value=observation),
                patch.object(env.game, "safety_score", return_value=0.8),
            ):
                return env.step(0)

        short_result = take_terminal_step(short_env)
        long_result = take_terminal_step(long_env)

        self.assertLess(
            long_result[4]["reward_death"],
            short_result[4]["reward_death"],
        )

    def test_timeout_uses_full_repeat_window(self):
        env = SnakeEnv(board_size=4)
        env.step_count = env.max_snake_length * 4 - 1
        observation = np.zeros(env.observation_space.shape, dtype=np.uint8)

        with (
            patch.object(env.game, "step", return_value=(False, 0)),
            patch.object(env, "_get_obs", return_value=observation),
            patch.object(env.game, "safety_score", return_value=0.8),
        ):
            _, _, terminated, truncated, info = env.step(0)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reward_timeout"], -1.0)

    def test_action_mask_disallows_reverse_direction(self):
        env = SnakeEnv(board_size=4)
        env.game.direction = "left"

        np.testing.assert_array_equal(
            env.mask_fn(),
            np.array([1, 1, 1, 0], dtype=np.int8),
        )

    def test_safety_score_detects_a_trapped_head(self):
        env = SnakeEnv(board_size=4)
        env.game.snake = deque([
            (1, 1),
            (0, 1),
            (0, 0),
            (1, 0),
            (2, 0),
            (2, 1),
            (2, 2),
            (1, 2),
            (0, 2),
            (0, 3),
        ])

        self.assertLess(env.game.safety_score(), 0.1)


if __name__ == "__main__":
    unittest.main()
