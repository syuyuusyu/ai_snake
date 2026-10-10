---
name: train-control
description: 在用户要求时调整正在运行的 ai_snake 训练的热参数，或停止训练（通过 train_ctl.py 写 runs/control.json）。用户说"调一下 X"、"把学习率降一点"、"停止训练"时使用。
model: haiku
---

# 调整或停止训练

所有操作**只能**通过 `train_ctl.py` 完成。它会校验参数和范围，写入是原子的，训练会在下一个 rollout 边界生效。

## 可以热调整的参数（范围以 agent_bridge.HOT_PARAMS 为准）

| 参数 | 范围 | 含义 |
|---|---|---|
| `ent_coef` | 0–0.02 | 熵奖励；绕圈超时多时可以小幅调高 |
| `target_kl` | 0.005–0.05 | PPO 每轮更新的 KL 上限 |
| `lr_scale` | 0.1–3.0 | 学习率倍率 |
| `clip_scale` | 0.5–2.0 | clip 倍率 |
| `bc_coef_start` | 0–1 | BC loss 权重 |
| `jev_budget` | 0–128 | 每批标注最多调用 Jev 的次数（`--no-jev` 时不可用） |
| `audit_rate` | 0–0.2 | 抽检比例（`--no-jev` 时不可用） |

## 流程

1. **每次只改一个参数。**
2. 如果用户已经说清了参数和数值，就直接执行；如果只说了个方向（比如"降一点"），先提议一个具体数值（建议幅度不超过当前值的 2 倍或 1/2），等用户确认后再执行。
3. 调整：
   ```bash
   .venv/bin/python train_ctl.py adjust <param> <value> --reason "<用户的原话或原因>"
   ```
4. 停止（会保存 `<save>_final.zip`，并正常退出、不会重启）：
   ```bash
   .venv/bin/python train_ctl.py stop --reason "<原因>"
   ```
5. 大约 1 分钟后跑一次 `.venv/bin/python train_ctl.py status`，确认最近的决策里出现了 `src=human ... applied=True`。

## 不在这张表里的改动

`n_steps`、`batch_size`、`gamma`、`gae_lambda`、奖励函数、网络结构、环境数量，都需要重启或者改代码。遇到这类请求：

- 向用户说明原因，并给出具体方案。例如："先 stop，再用 `/train-start --n-steps 1024 ...` 重新启动"。
- **不要自己改代码。**
- 用户明确同意"停止并用新参数重启"后，先执行 `stop`，等 `status` 显示 `not running`，再执行 `/train-start <新参数>`。

## 紧急情况

只有用户明确说"强制结束"或"kill"时，才执行：
```bash
.venv/bin/python train_ctl.py kill
```
这样不会保存 checkpoint，执行前要告诉用户这一点。
