"""
将 MaskablePPO 训练得到的 CnnPolicy 权重导出为 Swift / MLX 可对齐的格式，
并打印关键 shape 日志，方便排查 Swift 侧网络结构是否一致。

示例：
    uv run python convert_to_mlx.py --model pth/stable_4.zip --out pth/snake_policy.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO


def print_header(title: str) -> None:
    print(f"\n{'=' * 16} {title} {'=' * 16}")


def describe_array(name: str, value: np.ndarray) -> None:
    print(f"{name:<16} shape={tuple(value.shape)!s:<18} dtype={value.dtype}")


def export(model_path: str, out_path: str, board_size: int = 12, run_forward_check: bool = True) -> None:
    scale = max(1, (32 + board_size - 1) // board_size)
    shape_size = board_size * scale + 2 * scale

    print_header("CONFIG")
    print(f"model_path   : {model_path}")
    print(f"out_path     : {out_path}")
    print(f"board_size   : {board_size}")
    print(f"scale        : {scale}")
    print(f"shape_size   : {shape_size}")
    print(f"obs shape    : (3, {shape_size}, {shape_size})")

    print_header("LOAD MODEL")
    model = MaskablePPO.load(
        model_path,
        device="cpu",
        custom_objects={
            "observation_space": spaces.Box(
                low=0,
                high=255,
                shape=(3, shape_size, shape_size),
                dtype=np.uint8,
            ),
            "action_space": spaces.Discrete(4),
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )

    policy = model.policy
    print(f"policy class : {type(policy).__name__}")
    print(f"features ext : {type(policy.pi_features_extractor).__name__}")
    print(f"action net   : {type(policy.action_net).__name__}")

    sd = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}

    required_source_keys = [
        "pi_features_extractor.cnn.0.weight",
        "pi_features_extractor.cnn.0.bias",
        "pi_features_extractor.cnn.2.weight",
        "pi_features_extractor.cnn.2.bias",
        "pi_features_extractor.cnn.4.weight",
        "pi_features_extractor.cnn.4.bias",
        "pi_features_extractor.linear.0.weight",
        "pi_features_extractor.linear.0.bias",
        "action_net.weight",
        "action_net.bias",
    ]

    missing = [key for key in required_source_keys if key not in sd]
    if missing:
        raise KeyError(f"Missing expected state_dict keys: {missing}")

    print_header("SOURCE TENSOR SHAPES")
    for key in required_source_keys:
        describe_array(key, sd[key])

    if run_forward_check:
        print_header("FORWARD SHAPE CHECK")
        obs = torch.zeros(1, 3, shape_size, shape_size, dtype=torch.float32)
        with torch.no_grad():
            x = policy.pi_features_extractor.cnn(obs)
            print(f"cnn output shape      : {tuple(x.shape)}")

            flat = x.reshape(x.shape[0], -1)
            print(f"flattened shape       : {tuple(flat.shape)}")
            print(f"flattened dim         : {flat.shape[1]}")

            linear_out = policy.pi_features_extractor.linear(flat)
            print(f"linear output shape   : {tuple(linear_out.shape)}")

            logits = policy.action_net(linear_out)
            print(f"action logits shape   : {tuple(logits.shape)}")

        print("\nSwift 对齐提示：")
        print(f"- Conv2d(inputChannels:) 应为 3")
        print(f"- Linear(inputDimensions:) 应为 {flat.shape[1]}")
        print("- PyTorch Linear 权重 shape 是 (out, in)，MLX/Swift 侧通常需要转成 (in, out)")

    weights = {
        "cnn_0_w": sd["pi_features_extractor.cnn.0.weight"],
        "cnn_0_b": sd["pi_features_extractor.cnn.0.bias"],
        "cnn_2_w": sd["pi_features_extractor.cnn.2.weight"],
        "cnn_2_b": sd["pi_features_extractor.cnn.2.bias"],
        "cnn_4_w": sd["pi_features_extractor.cnn.4.weight"],
        "cnn_4_b": sd["pi_features_extractor.cnn.4.bias"],
        "linear_w": sd["pi_features_extractor.linear.0.weight"],
        "linear_b": sd["pi_features_extractor.linear.0.bias"],
        "action_w": sd["action_net.weight"],
        "action_b": sd["action_net.bias"],
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print_header("EXPORTED TENSOR SHAPES")
    for key, value in weights.items():
        describe_array(key, value)

    suffix = out_file.suffix.lower()
    if suffix == ".safetensors":
        mlx_weights = {
            key: mx.array(np.ascontiguousarray(value), dtype=mx.float32)
            for key, value in weights.items()
        }
        mx.save_safetensors(str(out_file), mlx_weights)
    else:
        raise ValueError("Unsupported output extension. Use .safetensors")

    print_header("DONE")
    print(f"Saved policy weights to {out_file}")
    print(f"Exported keys: {sorted(weights.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pth/stable_4.zip")
    parser.add_argument("--out", default="pth/snake_policy.safetensors")
    parser.add_argument("--board-size", type=int, default=12)
    parser.add_argument(
        "--skip-forward-check",
        action="store_true",
        help="Skip the dummy forward pass shape diagnostics.",
    )
    args = parser.parse_args()

    export(
        model_path=args.model,
        out_path=args.out,
        board_size=args.board_size,
        run_forward_check=not args.skip_forward_check,
    )


if __name__ == "__main__":
    main()

