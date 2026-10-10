"""Play one SnakeEnv episode with TypeSafe Jev choosing safe directions.

Requests go through the local OpenRouter proxy. Use --preview-state to inspect
the exact request without making an API call.
"""

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from board_config import add_board_arguments
from game_env import SnakeEnv

API_URL = "https://www.ufoyy.top:7022/invoke/jev"
DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def build_state(game):
    """Represent the ordered snake and the environment's tail BFS area."""
    snake = list(game.snake)
    head = snake[0]
    board_width, board_high = game.board_width, game.board_high
    rows = [["."] * board_width for _ in range(board_high)]
    for index, (x, y) in enumerate(snake):
        rows[y][x] = index
    food_x, food_y = game.food
    rows[food_y][food_x] = "F"
    tail_bfs_area = game.bfs_reachable_area()
    for x, y in tail_bfs_area:
        if rows[y][x] == ".":
            rows[y][x] = "S"

    moves = {}
    for direction in game.directions:
        if direction == OPPOSITE[game.direction]:
            continue
        dx, dy = DELTAS[direction]
        target = (head[0] + dx, head[1] + dy)
        if not (0 <= target[0] < board_width and 0 <= target[1] < board_high):
            continue
        eats_food = target == game.food
        occupied = snake if eats_food else snake[:-1]
        if target in occupied:
            continue
        moves[direction] = {
            "target": list(target),
            "eats_food": eats_food,
            "wins": eats_food and len(snake) + 1 == board_width * board_high,
            "food_distance": abs(target[0] - food_x) + abs(target[1] - food_y),
            "target_in_tail_bfs": target in tail_bfs_area,
        }

    return {
        "board_width": board_width,
        "board_high": board_high,
        "coordinates": "[x,y], x increases right, y increases down",
        "board_legend": "0=head; larger integers are closer to tail. S=empty cell in the current tail-connected BFS region; .=empty cell outside that region; F=food. S does not guarantee future safety. Rows go top to bottom; on a non-food move the old tail leaves before the new head enters, and the old head becomes body.",
        "board_rows": rows,
        "snake_length": len(snake),
        "free_cells": board_width * board_high - len(snake),
        "direction": game.direction,
        "food": list(game.food),
        "safe_moves": moves,
    }


def build_request(state, model):
    moves = state["safe_moves"]
    return {
        "model": model,
        "state": state,
        "questions": {
            "move": {
                "type": "choice",
                "instructions": (
                    "Which safe move best preserves future mobility while progressing toward food? "
                    "Prefer an immediate win, then continued survival over shorter food distance. "
                    "S marks the current tail-connected region; . marks empty cells outside it. "
                    "Before choosing, account for the old head becoming body and the tail vacating. "
                    "Reject entry into a one-cell pocket that leaves no legal next move. "
                    "For disconnected regions, check whether an exit can open before space runs out. "
                    "All listed moves survive only this step; neither S nor . guarantees future safety."
                ),
                "criteria": {
                    direction: (
                        f"Move {direction} to {details['target']}; target is "
                        f"{'inside' if details['target_in_tail_bfs'] else 'outside'} "
                        "the current tail-connected region"
                    )
                    for direction, details in moves.items()
                },
            }
        },
    }


def ask_jev(request_body, max_attempts=5):
    request = Request(
        API_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "logKey": "step"
            },
        method="POST",
    )
    retryable_codes = {429, 500, 502, 503, 504, 529}
    for attempt in range(max_attempts):
        try:
            with urlopen(request, timeout=30) as response:
                result = json.load(response)
            if "error" in result:
                error = result["error"]
                if isinstance(error, dict) and error.get("code") in retryable_codes and attempt + 1 < max_attempts:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f"Local Jev proxy error: {result['error']}")
            try:
                answer = result["answers"]["move"]
                direction = answer["choice"]
            except (KeyError, TypeError):
                if attempt + 1 < max_attempts:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f"Local Jev proxy returned no move answer: {result}")
            if direction not in request_body["state"]["safe_moves"]:
                raise ValueError(f"Jev returned an unavailable direction: {direction!r}")
            return direction, answer.get("confidence")
        except HTTPError as error:
            if error.code in retryable_codes and attempt + 1 < max_attempts:
                time.sleep(2**attempt)
                continue
            detail = error.read(500).decode("utf-8", errors="replace")
            raise RuntimeError(f"Local Jev proxy returned HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            if attempt + 1 < max_attempts:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(
                f"Local Jev proxy request failed: {getattr(error, 'reason', error)}"
            ) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_board_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="typesafe/jev-1.13", help="OpenRouter model ID")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--render", action="store_true", help="Show the pygame board")
    parser.add_argument("--preview-state", action="store_true", help="Print one request without calling Jev")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be >= 1")

    env = SnakeEnv(board_width=args.board_width, board_high=args.board_high, silent_mode=not args.render, seed=args.seed)
    try:
        env.reset()
        for step in range(1, args.max_steps + 1):
            if args.render:
                import pygame

                if any(event.type == pygame.QUIT for event in pygame.event.get()):
                    print("Window closed.")
                    return
                env.render()

            state = build_state(env.game)
            request_body = build_request(state, args.model)
            if args.preview_state:
                print(json.dumps(request_body, indent=2))
                return

            safe_moves = state["safe_moves"]
            winning_moves = [name for name, details in safe_moves.items() if details["wins"]]
            if winning_moves:
                direction, confidence = winning_moves[0], None
            elif len(safe_moves) > 1:
                direction, confidence = ask_jev(request_body)
            elif safe_moves:
                direction, confidence = next(iter(safe_moves)), None
            else:
                direction = next(
                    candidate for candidate in env.game.directions
                    if candidate != OPPOSITE[env.game.direction]
                )
                confidence = None

            _, _, terminated, truncated, info = env.step(env.game.directions.index(direction))
            if info["step_state"] == "eat food":
                print(f"step={step} length={info['snake_length']} ate food move={direction} confidence={confidence}")
            if terminated or truncated:
                print(
                    f"result={info['step_state'] if terminated else 'timeout'} "
                    f"steps={step} length={info['snake_length']} move={direction}"
                )
                return
        print(
            f"result=max_steps steps={args.max_steps} length={len(env.game.snake)} "
            f"last_move={direction} confidence={confidence}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
