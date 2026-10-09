import io
import json
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from play_with_jev import ask_jev, build_request, build_state
from snake_game import SnakeGame


class PlayWithJevTest(unittest.TestCase):
    def game(self, snake, direction, food, board_size=4):
        game = SimpleNamespace(
            snake=deque(snake),
            direction=direction,
            food=food,
            board_size=board_size,
            directions=["up", "down", "left", "right"],
        )
        game.bfs_reachable_area = SnakeGame.bfs_reachable_area.__get__(game)
        return game

    def test_state_uses_ordered_board_and_filters_reverse_and_collisions(self):
        game = self.game([(1, 1), (2, 1), (3, 1)], "left", (0, 1))

        state = build_state(game)

        self.assertEqual(state["board_rows"][1], ["F", 0, 1, 2])
        self.assertNotIn("snake_head_to_tail", state)
        self.assertNotIn("right", state["safe_moves"])
        self.assertTrue(state["safe_moves"]["left"]["eats_food"])
        self.assertEqual(state["safe_moves"]["left"]["food_distance"], 0)

    def test_ordered_board_distinguishes_snakes_with_same_occupied_cells(self):
        first = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1),
                 (0, 2), (1, 2), (2, 2), (3, 2), (3, 1), (3, 0)]
        second = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2),
                  (2, 2), (3, 2), (3, 1), (2, 1), (2, 0), (3, 0)]
        self.assertEqual(set(first), set(second))

        first_rows = build_state(self.game(first, "left", (0, 3)))["board_rows"]
        second_rows = build_state(self.game(second, "left", (0, 3)))["board_rows"]

        self.assertNotEqual(first_rows, second_rows)
        self.assertEqual(first_rows[0][2], 2)
        self.assertEqual(second_rows[0][2], 10)

    def test_environment_tail_bfs_marks_board_without_simulating_moves(self):
        board = [
            [".", 19, 20, 21, 50, 51, 52, 53, 54, 55, ".", "."],
            [".", 18, 23, 22, 49, 48, ".", ".", ".", 56, ".", "."],
            [".", 17, 24, 31, 32, 47, ".", ".", ".", 57, ".", "."],
            [".", 16, 25, 30, 33, 46, ".", ".", ".", 58, ".", "."],
            [".", 15, 26, 29, 34, 45, ".", ".", ".", 59, ".", "."],
            [".", 14, 27, 28, 35, 44, ".", ".", ".", 60, ".", "."],
            [".", 13, 38, 37, 36, 43, ".", ".", ".", ".", ".", "."],
            [".", 12, 39, 40, 41, 42, ".", ".", ".", ".", ".", "."],
            [".", 11, 10, 9, 8, 7, ".", ".", ".", ".", ".", "."],
            [".", "F", ".", ".", ".", 6, 5, 4, 3, 2, 1, 0],
            ["."] * 12,
            ["."] * 12,
        ]
        snake = [None] * 61
        for y, row in enumerate(board):
            for x, cell in enumerate(row):
                if isinstance(cell, int):
                    snake[cell] = (x, y)

        state = build_state(self.game(snake, "right", (1, 9), board_size=12))

        self.assertEqual(set(state["safe_moves"]), {"up", "down"})
        self.assertEqual(state["board_rows"][2][8], "S")
        self.assertEqual(state["board_rows"][9][1], "F")
        self.assertEqual(state["board_rows"][10][1], ".")
        self.assertEqual(state["board_rows"][9][11], 0)
        self.assertEqual(state["board_rows"][5][9], 60)
        self.assertNotIn("reachable_cells", state["safe_moves"]["up"])
        self.assertNotIn("tail_reachable", state["safe_moves"]["down"])

    def test_tail_bfs_does_not_mark_other_side_of_body(self):
        snake = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (0, 4)]

        state = build_state(self.game(snake, "left", (4, 4), board_size=5))

        self.assertEqual(state["board_rows"][0][0], "S")
        self.assertEqual(state["board_rows"][2][2], ".")
        self.assertEqual(state["board_rows"][4][4], "F")
        self.assertEqual(state["board_rows"][4][0], 5)

    def test_one_cell_pocket_is_outside_tail_region_and_has_no_exit(self):
        snake = [(7, 6), (6, 6), (6, 7), (6, 8), (7, 8), (8, 8),
                 (8, 7), (9, 7), (9, 6), (9, 5), (9, 4), (9, 3),
                 (9, 2), (9, 1), (9, 0), (8, 0)]
        game = self.game(snake, "right", (11, 11), board_size=12)
        state = build_state(game)
        request = build_request(state, "typesafe/jev-1.13")

        self.assertEqual(state["board_rows"][7][7], ".")
        self.assertFalse(state["safe_moves"]["down"]["target_in_tail_bfs"])
        self.assertTrue(state["safe_moves"]["up"]["target_in_tail_bfs"])
        self.assertTrue(state["safe_moves"]["right"]["target_in_tail_bfs"])
        self.assertIn("outside", request["questions"]["move"]["criteria"]["down"])
        game.direction = "down"
        terminated, _ = SnakeGame.step(game)
        self.assertFalse(terminated)
        self.assertEqual(build_state(game)["safe_moves"], {})

    def test_moving_into_current_tail_is_safe(self):
        game = self.game([(1, 1), (1, 2), (2, 2), (2, 1)], "up", (0, 0))

        state = build_state(game)

        self.assertIn("right", state["safe_moves"])
        self.assertEqual(state["safe_moves"]["right"]["target"], [2, 1])

    def test_final_food_is_marked_as_win(self):
        game = self.game([(0, 0), (1, 0), (1, 1)], "left", (0, 1), board_size=2)

        state = build_state(game)

        self.assertTrue(state["safe_moves"]["down"]["wins"])

    def test_request_and_response_use_choice_options(self):
        state = build_state(self.game([(1, 1), (2, 1), (3, 1)], "left", (0, 1)))
        request = build_request(state, "typesafe/jev-1.13")
        response = io.BytesIO(json.dumps({
            "model": "typesafe/jev-1.13-20260917",
            "answers": {"move": {"type": "choice", "choice": "left", "confidence": 0.8}},
        }).encode())

        with (
            patch("play_with_jev.urlopen", return_value=response) as mock_urlopen,
            patch("builtins.print") as mock_print,
        ):
            direction, confidence = ask_jev(request)

        self.assertEqual((direction, confidence), ("left", 0.8))
        self.assertIsInstance(request["state"], dict)
        mock_print.assert_not_called()
        sent_request = mock_urlopen.call_args.args[0]
        self.assertEqual(sent_request.full_url, "https://www.ufoyy.top:7022/invoke/jev")
        self.assertIsNone(sent_request.get_header("Authorization"))
        self.assertEqual(json.loads(sent_request.data), request)

    def test_rejects_direction_not_in_safe_moves(self):
        state = build_state(self.game([(1, 1), (2, 1), (3, 1)], "left", (0, 1)))
        response = io.BytesIO(b'{"answers":{"move":{"choice":"right"}}}')

        with patch("play_with_jev.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "unavailable direction"):
                ask_jev(build_request(state, "typesafe/jev-1.13"))

    def test_rejects_proxy_error_inside_http_200(self):
        state = build_state(self.game([(1, 1), (2, 1), (3, 1)], "left", (0, 1)))
        response = io.BytesIO(b'{"error":{"message":"invalid model","code":400}}')

        with patch("play_with_jev.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "invalid model"):
                ask_jev(build_request(state, "typesafe/jev-1.13"))

    def test_retries_transient_proxy_responses(self):
        state = build_state(self.game([(1, 1), (2, 1), (3, 1)], "left", (0, 1)))
        responses = [
            io.BytesIO(b'{"error":{"message":"unavailable","code":503}}'),
            io.BytesIO(b'{"model":"typesafe/jev-1.13"}'),
            io.BytesIO(b'{"answers":{"move":{"choice":"left"}}}'),
        ]
        with (
            patch("play_with_jev.urlopen", side_effect=responses) as urlopen,
            patch("play_with_jev.time.sleep") as sleep,
        ):
            direction, _ = ask_jev(build_request(state, "typesafe/jev-1.13"))

        self.assertEqual(direction, "left")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_retries_request_timeout(self):
        state = build_state(self.game([(1, 1), (2, 1), (3, 1)], "left", (0, 1)))
        responses = [
            TimeoutError("timed out"),
            io.BytesIO(b'{"answers":{"move":{"choice":"left"}}}'),
        ]
        with (
            patch("play_with_jev.urlopen", side_effect=responses),
            patch("play_with_jev.time.sleep") as sleep,
        ):
            direction, _ = ask_jev(build_request(state, "typesafe/jev-1.13"))

        self.assertEqual(direction, "left")
        sleep.assert_called_once()

    def test_model_is_sent(self):
        state = build_state(self.game([(1, 1), (2, 1), (3, 1)], "left", (0, 1)))

        self.assertEqual(build_request(state, "typesafe/jev-1.13")["model"], "typesafe/jev-1.13")


if __name__ == "__main__":
    unittest.main()
