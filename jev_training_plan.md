# 将 Jev 接入训练，并尽量减少 Jev 调用次数

> 背景：`pth/stable_4.zip` 是目前最好的模型，继续训练没有实质提升。现在引入 Jev 辅助训练，目标是把 Jev 用进训练流程，同时尽可能少调用它。
> 日期：2026-10-09

## 核心结论

**Jev 不要当策略用，也不要每步都当老师。** 只在极少数"关键状态"上调用，而且排在两层免费检查之后。PPO 照常 on-policy 训练，Jev 给出的标签通过一个辅助 BC loss 用进去。

## 基线数据（种子 0–49，12×12，确定性策略）

和之前 Jev 评估用的是同一批种子。

| 方案 | 胜局 | 平均最长 | 中位数 | Jev/顾问调用 |
|---|---:|---:|---:|---:|
| Jev 1.13 单独玩 | 0/50 | 52.6 | 53 | 每个多选步都调 |
| stable_4 原样 | 2/50 | 86.5 | 109.5 | 0 |
| stable_4 + 屏蔽立即致死动作 | **13/50** | 101.6 | 129 | **0** |
| 再加：只在门控点问"完美顾问" | **25/50** | 128.4 | 143.5 | **平均 6.7 次/局（0.28% 的步）** |

关键发现：

- **Jev 单独玩比 stable_4 弱很多**，所以它不适合当老师逐步模仿，只适合在 PPO 判断不了的少数点上当裁判。
- stable_4 的 41 次死亡里，**有 29 次当时明明有安全步**，模型却几乎 100% 选了致死那一步。现在的 `mask_fn` 只屏蔽掉头，把致死动作也屏蔽掉就能修，不需要 Jev。
- 7 次超时是开局阶段（蛇长 4–43）在原地绕圈，也不需要 Jev。
- "门控点"指的是：PPO 选的那步会切断蛇头到蛇尾的连通，而另一个安全步不会。这种点每局只出现约 5 次，而且死亡几乎都紧跟在它后面。
- 光靠规则"永远不切断尾巴"也不行，蛇会一直追着尾巴转，最后超时。所以这些点确实需要更聪明的判断，这才是 Jev 该出场的地方。
- 详细数据见 `evaluations/stable_4_gating/`。

## 调用漏斗：便宜的检查先做，Jev 放最后

| 层 | 做什么 | Jev 调用 |
|---|---|---|
| L0 | 屏蔽致死动作（训练和推理都加） | 0 |
| L1 | BFS 门控：`risk`（切断尾巴）和 `loop`（同一状态重复出现），其余 99.7% 的步直接放过 | 0 |
| L2 | 按棋盘对称归一化后查缓存（正方形 8 种，矩形 4 种），命中直接用 | 0 |
| L3 | 用当前策略从每个安全步往后推演 200 步。有的分支死、有的分支活，或者分数差至少 1 个食物，就直接定答案 | 0 |
| L4 | 推演分不出高下，才调 Jev。并发请求，每轮最多 `--jev-budget` 次（默认 32） | 1 |
| 抽检 | L3 已经定了答案的状态，再按 5% 抽样问一次 Jev，统计它和推演结论的一致率 | 少量 |

**调用量估算**：原来的做法是每个多选步都问 Jev，约占 84% 的步，训练 500 万步大约要 420 万次。按上面的设计，500 万步约 152 轮，每轮最多 32 次，**上限大约 4,900 次**，缓存命中后还会更少。

## 怎么放进训练

- **Rollout 时不让 Jev 改动作。** 如果直接替换动作，PPO 会用新策略对这个动作的概率去算重要性比率，Jev 选的动作概率常常接近 0，比率会爆炸；而且奖励会记到不是策略自己选的动作上。
- **改成 DAgger 式的辅助 BC loss：**
  1. Env wrapper 在门控点记录下当时的状态（不改动作）。
  2. 每轮 rollout 结束后，在后台线程里标注这些状态，不阻塞 PPO 更新。
  3. 每次 `train()` 之后，在标签池上加一个 `λ·CE(π(s), label)`，λ 从 0.5 线性衰减到 0.05。
  4. 推演验证过的标签权重 1.0；Jev 的标签权重 0.5，因为它自己玩得不如 PPO。
- 每个标签会在很多轮里反复用到，这是调用效率的主要来源。

## 风险与注意事项

- **25/50 那一行用的不是 Jev**，是 PPO 自己的推演结果，也就是说一次 Jev 都没调就从 13 胜到了 25 胜。Jev 到底能比推演多带来多少，**目前没有证据**。
- 所以先跑 `--no-jev` 作对照（只用推演标签），再开 Jev，同时看 `jev/audit_agree_rate`。如果 Jev 打不过对照，就没必要调它。
- 25 胜是推理时实时问顾问的结果。通过 BC 学进权重后，未必能保留这么多。屏蔽致死动作加门控本身很便宜，推理时可以一直开着。
- 剩下的失败主要在残局（蛇长 127–143）。而 Jev 自己平均只能活到 52 左右，它在残局能帮多少要打个问号。
- 开局绕圈可以用免费规则解决（检测到循环时，在保持尾巴可达的前提下朝食物走），不需要 Jev。

## 相关文件

| 文件 | 内容 |
|---|---|
| `jev_assist.py` | 安全 mask、门控、对称缓存、标注器（推演 → Jev → 抽检）和标签池 |
| `train_with_jev.py` | 从 stable_4 继续训练，带 BC loss；最佳模型存到 `pth/stable_7_jev_best` |
| `simulate_gates.py` | 复现上面的对比实验：`python simulate_gates.py {safe\|tail\|oracle} 50 out.jsonl 0` |
| `evaluations/stable_4_gating/` | 上面数据的说明和逐局结果 |
| `tests/test_jev_assist.py` | 安全 mask、门控及对称缓存的基础测试 |
| `tests/test_jev_training.py` | 调用预算、缓存来源、并发隔离、BC 更新、保存及重新加载测试 |

验证状态：已通过模拟 Jev 测试及强制门控的 PPO + 后台标注 + BC 集成测试；从 `stable_4` 完成了 1,024 步 `--no-jev` 流程验证，检查了评估、日志、最佳/最终模型保存。该短运行没有触发门控，BC 路径由集成测试验证。真实 Jev 接口和完整 500 万步训练尚未运行；没有新的效果结论。

## 运行步骤

```bash
# 1. 对照组：只用推演标签，0 次 Jev 调用
.venv/bin/python train_with_jev.py --no-jev --save pth/stable_7_ctrl

# 2. 实验组：开启 Jev，每轮最多 32 次
.venv/bin/python train_with_jev.py --jev-budget 32 --save pth/stable_7_jev
```

对比两组的 `eval/score`、胜局数，以及实验组的 `jev/jev_calls`、`jev/audit_agree_rate`，再决定是否保留 Jev。


## 实现与配置

- 棋盘默认值统一取自 `board_config.py`，为 `board_width=12`、`board_high=12`；入口的 `--board-width`、`--board-high` 会传到训练、推演、缓存和评估环境。
- 每个环境使用独立的食物随机数发生器。快照同时保存随机数状态和距上次吃食物的步数，保证各推演分支从同一状态开始，后台标注不改变训练环境。
- 缓存记录标签来源，推演标签权重保持 1.0，Jev 标签保持 0.5。键包含宽高、方向及超时步数。旧缓存中没有来源的记录不会复用。
- `--no-jev` 不发送请求，也不使用 Jev 来源的缓存标签；默认使用 `evaluations/ctrl_cache.jsonl`，实验组默认使用 `evaluations/jev_cache.jsonl`。
- 训练请求每次只尝试一次，网络失败跳过该标签，不自动重试。`--jev-budget` 包含抽检，因此不会因隐藏重试突破请求预算；独立 Jev 游玩脚本保留原有重试行为。
- 每次只允许一个后台标注批次使用冻结策略；后台忙时丢弃新快照并记录 `jev/dropped_busy`，避免积压。退出前等待最后一批标签并执行可用的 BC 更新，再做最终评估和保存。
- 默认评估种子为 0–49，每 500,000 个环境步评估；支持 `--eval-episodes`、`--eval-max-steps` 调整，`--eval-freq 0` 关闭评估。
- 输出包含 `<save>_best.zip`（启用评估时）、`<save>_final.zip`；`logs/<save 文件名>/` 保存 TensorBoard、`progress.csv`、`config.json` 和 `jev_stats.json`，可用 `--log-dir` 修改。
- 最终模型不包含内存标签池，可以用标准 `MaskablePPO.load()` 加载。推理入口已使用相同的致死动作 mask：

```bash
.venv/bin/python play_with_model.py --model pth/stable_7_jev_best.zip
```

快速验证命令（只验证流程，不代表训练效果；输出写到 `/tmp`）：

```bash
.venv/bin/python train_with_jev.py --no-jev --device cpu \
  --envs 1 --timesteps 1024 --n-steps 32 --batch-size 32 \
  --horizon 8 --max-labels 8 --bc-min-size 1 --bc-batch-size 8 --bc-steps 1 \
  --eval-freq 512 --eval-episodes 2 --eval-max-steps 64 \
  --save /tmp/snake-jev-smoke/model --cache /tmp/snake-jev-smoke/cache.jsonl \
  --log-dir /tmp/snake-jev-smoke/logs
```
