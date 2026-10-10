"""Evaluate PPO checkpoints on a fixed seed range, with or without the move shield.

Training evaluates on seeds 0-49, so use another range (default 1000-1199) to compare models fairly.

  python evaluate_model.py pth/stable_4.zip
  python evaluate_model.py pth/a.zip pth/b.zip --seeds 1000:1200 --no-shield
"""

import argparse
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from board_config import DEFAULT_BOARD_HIGH, DEFAULT_BOARD_WIDTH, add_board_arguments


def evaluate(path, seeds, shield, board_width=DEFAULT_BOARD_WIDTH, board_high=DEFAULT_BOARD_HIGH,
             max_steps=20_000):
    from sb3_contrib import MaskablePPO

    import train_with_ppo as base
    from jev_assist import safe_mask_fn
    from train_with_jev import SafeSnakeEnv

    env = SafeSnakeEnv(seed=seeds[0], board_width=board_width, board_high=board_high, silent_mode=True,
                       bfs_intensity=base.bfs_intensity, reward_gamma=base.discount_gamma)
    mask = env.mask_fn if shield else (lambda: safe_mask_fn(env))
    const = lambda _: 0.0  # noqa: E731
    model = MaskablePPO.load(path, device="cpu", custom_objects={
        "observation_space": env.observation_space, "action_space": env.action_space,
        "learning_rate": const, "lr_schedule": const, "clip_range": const})

    scores, causes = [], Counter()
    for seed in seeds:
        random.seed(seed)
        obs, _ = env.reset(seed=seed)
        best, cause = len(env.game.snake), "step_cap"
        for _ in range(max_steps):
            action, _ = model.predict(obs, action_masks=mask(), deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            best = max(best, int(info["snake_length"]))
            if terminated or truncated:
                cause = info["step_state"] if terminated else "timeout"
                break
        causes[cause] += 1
        # Same scoring as FixedSeedEvalCallback: max length, plus board area for a win.
        scores.append(best + (env.max_snake_length if cause == "Victory" else 0))
    env.close()
    n, scores = len(seeds), np.asarray(scores, dtype=np.float64)
    return {
        "model": path,
        "score": round(float(scores.mean()), 1),
        "score_se": round(float(scores.std(ddof=1) / np.sqrt(n)), 1) if n > 1 else 0.0,
        "win": causes["Victory"] / n,
        "timeout": causes["timeout"] / n,
        "death": (causes["hit wall"] + causes["collied self"]) / n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seeds", default="1000:1200", help="start:stop range of episode seeds")
    ap.add_argument("--no-shield", action="store_true", help="only mask immediately lethal moves")
    ap.add_argument("--workers", type=int, default=4)
    add_board_arguments(ap)
    args = ap.parse_args()
    start, stop = map(int, args.seeds.split(":"))
    seeds = list(range(start, stop))
    jobs = [(m, seeds, not args.no_shield, args.board_width, args.board_high) for m in args.models]
    print(f"seeds {start}-{stop - 1}, shield {'off' if args.no_shield else 'on'}")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for r in ex.map(evaluate, *zip(*jobs)):
            print(f"{r['model']}: score={r['score']}±{r['score_se']} win={r['win']:.1%} "
                  f"timeout={r['timeout']:.1%} death={r['death']:.1%}")


if __name__ == "__main__":
    main()
