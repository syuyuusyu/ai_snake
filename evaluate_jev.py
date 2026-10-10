"""Evaluate Jev on fixed Snake seeds and save the steps before each failure.

Example: .venv/bin/python evaluate_jev.py --seeds 50 --output-dir /tmp/jev-eval
"""

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from statistics import mean, median

from board_config import add_board_arguments
from game_env import SnakeEnv
from play_with_jev import OPPOSITE, ask_jev, build_request, build_state


def choose_move(game, state, model):
    safe_moves = state["safe_moves"]
    winning_moves = [move for move, details in safe_moves.items() if details["wins"]]
    if winning_moves:
        return winning_moves[0], None, "immediate_win"
    if len(safe_moves) > 1:
        move, confidence = ask_jev(build_request(state, model))
        return move, confidence, "jev"
    if safe_moves:
        return next(iter(safe_moves)), None, "only_safe_move"
    move = next(move for move in game.directions if move != OPPOSITE[game.direction])
    return move, None, "no_safe_move"


def run_episode(seed, board_width, board_high, model, max_steps, trace_steps):
    env = SnakeEnv(board_width=board_width, board_high=board_high, silent_mode=True, seed=seed)
    history = deque(maxlen=trace_steps)
    max_length = len(env.game.snake)
    try:
        env.reset()
        for step in range(1, max_steps + 1):
            state = build_state(env.game)
            move, confidence, source = choose_move(env.game, state, model)
            _, _, terminated, truncated, info = env.step(env.game.directions.index(move))
            max_length = max(max_length, state["snake_length"], info["snake_length"])
            history.append({
                "step": step,
                "state": state,
                "chosen_move": move,
                "choice_source": source,
                "confidence": confidence,
                "step_state": info["step_state"],
                "resulting_length": info["snake_length"],
            })
            if terminated or truncated:
                outcome = (
                    "victory" if info["step_state"] == "Victory"
                    else "death" if terminated else "timeout"
                )
                break
        else:
            outcome = "max_steps"
            info = {"snake_length": len(env.game.snake), "step_state": None}

        result = {
            "seed": seed,
            "outcome": outcome,
            "reason": info["step_state"] if outcome == "death" else outcome,
            "steps": step,
            "final_length": info["snake_length"],
            "max_length": max_length,
        }
        return result, list(history) if outcome != "victory" else []
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=50, help="Number of consecutive seeds")
    parser.add_argument("--start-seed", type=int, default=0)
    add_board_arguments(parser)
    parser.add_argument("--model", default="typesafe/jev-1.13")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--trace-steps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.max_steps < 1 or args.trace_steps < 1:
        parser.error("--seeds, --max-steps and --trace-steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty to keep runs separate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = args.output_dir / "failures"
    failures_dir.mkdir(exist_ok=True)

    results = []
    with (args.output_dir / "episodes.jsonl").open("w") as episodes_file:
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            result, trace = run_episode(
                seed, args.board_width, args.board_high, args.model, args.max_steps, args.trace_steps
            )
            results.append(result)
            episodes_file.write(json.dumps(result) + "\n")
            episodes_file.flush()
            if trace:
                with (failures_dir / f"seed_{seed}.json").open("w") as trace_file:
                    json.dump({"result": result, "last_steps": trace}, trace_file, indent=2)
            print(
                f"seed={seed} outcome={result['outcome']} reason={result['reason']} "
                f"steps={result['steps']} max_length={result['max_length']}",
                flush=True,
            )

    outcomes = Counter(result["outcome"] for result in results)
    summary = {
        "model": args.model,
        "board_width": args.board_width,
        "board_high": args.board_high,
        "start_seed": args.start_seed,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "trace_steps": args.trace_steps,
        "victories": outcomes["victory"],
        "win_rate": outcomes["victory"] / len(results),
        "outcomes": dict(outcomes),
        "death_reasons": dict(Counter(
            result["reason"] for result in results if result["outcome"] == "death"
        )),
        "maximum_length": max(result["max_length"] for result in results),
        "mean_max_length": mean(result["max_length"] for result in results),
        "median_max_length": median(result["max_length"] for result in results),
    }
    with (args.output_dir / "summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
