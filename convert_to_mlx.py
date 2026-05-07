"""
将 MaskablePPO 训练的 CnnPolicy 权重导出为 MLX 可加载的 NPZ 格式。

用法：
    conda run -n ai_snake python convert_to_mlx.py --model pth/stable_4.zip --out pth/snake_policy.npz

推理时只需要 policy 路径（pi_features_extractor + action_net）：
    obs -> pi_features_extractor -> action_net -> logits -> mask -> argmax
"""

import argparse
import numpy as np
from sb3_contrib import MaskablePPO


def export(model_path: str, out_path: str) -> None:
    print(f"Loading model from {model_path} ...")
    model = MaskablePPO.load(model_path, device="cpu")
    policy = model.policy
    sd = {k: v.cpu().numpy() for k, v in policy.state_dict().items()}

    # Policy inference path: pi_features_extractor -> action_net
    weights = {
        # CNN layers (named to match mlx_snake_policy.py)
        "cnn_0_w": sd["pi_features_extractor.cnn.0.weight"],   # (32, 3, 8, 8)
        "cnn_0_b": sd["pi_features_extractor.cnn.0.bias"],     # (32,)
        "cnn_2_w": sd["pi_features_extractor.cnn.2.weight"],   # (64, 32, 4, 4)
        "cnn_2_b": sd["pi_features_extractor.cnn.2.bias"],     # (64,)
        "cnn_4_w": sd["pi_features_extractor.cnn.4.weight"],   # (64, 64, 3, 3)
        "cnn_4_b": sd["pi_features_extractor.cnn.4.bias"],     # (64,)
        # Linear after CNN: (512, 64) -> (64, 512) after transpose for MLX
        "linear_w": sd["pi_features_extractor.linear.0.weight"],  # (512, 64)
        "linear_b": sd["pi_features_extractor.linear.0.bias"],    # (512,)
        # Action head
        "action_w": sd["action_net.weight"],  # (4, 512)
        "action_b": sd["action_net.bias"],    # (4,)
    }

    np.savez(out_path, **weights)
    print(f"Saved policy weights to {out_path}")
    for k, v in weights.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pth/stable_4.zip")
    parser.add_argument("--out", default="pth/snake_policy.npz")
    args = parser.parse_args()
    export(args.model, args.out)
