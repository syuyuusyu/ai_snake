import unittest
from unittest.mock import patch

from evaluate_jev import choose_move, run_episode


class EvaluateJevTest(unittest.TestCase):
    def test_choose_move_calls_jev_only_when_there_is_a_choice(self):
        class Game:
            directions = ["up", "down", "left", "right"]
            direction = "left"

        game = Game()
        with patch("evaluate_jev.ask_jev", return_value=("up", 0.8)) as ask:
            move = choose_move(game, {"safe_moves": {
                "up": {"wins": False, "target": [1, 0], "target_in_tail_bfs": True},
                "left": {"wins": False, "target": [0, 1], "target_in_tail_bfs": True},
            }}, "typesafe/jev-1.13")
            self.assertEqual(move, ("up", 0.8, "jev"))
            ask.assert_called_once()

            ask.reset_mock()
            move = choose_move(game, {"safe_moves": {
                "up": {"wins": True, "target": [1, 0]},
                "left": {"wins": False, "target": [0, 1]},
            }}, "typesafe/jev-1.13")
            self.assertEqual(move, ("up", None, "immediate_win"))
            ask.assert_not_called()

    def test_episode_trace_contains_full_pre_move_state(self):
        def first_choice(request):
            return next(iter(request["state"]["safe_moves"])), 0.5

        with patch("evaluate_jev.ask_jev", side_effect=first_choice):
            result, trace = run_episode(7, 7, 7, "typesafe/jev-1.13", 2, 20)

        self.assertEqual(result["seed"], 7)
        self.assertEqual(result["outcome"], "max_steps")
        self.assertEqual(result["steps"], 2)
        self.assertEqual(len(trace), 2)
        self.assertIn("board_rows", trace[0]["state"])
        self.assertNotIn("snake_head_to_tail", trace[0]["state"])
        self.assertIn("safe_moves", trace[0]["state"])
        self.assertIn(trace[0]["chosen_move"], trace[0]["state"]["safe_moves"])


if __name__ == "__main__":
    unittest.main()
