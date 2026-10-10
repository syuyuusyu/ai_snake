# 用 Claude 监督训练（agent supervision）

> 配套文件：`agent_bridge.py`、`supervisor.py`、`train_with_jev.py --agent`、`tests/test_agent_bridge.py`
> Jev 训练方案本身见 `jev_training_plan.md`。

## 分层：便宜的先做，模型最后

| 层 | 谁负责 | 模型调用 |
|---|---|---|
| 判断某次评估值不值得问 | `AgentBridgeCallback.triggers_for` 里的规则 | 0 |
| 给出一个决策 | `claude -p`：只读权限，按 JSON schema 输出 | 每次运行最多 `--agent-max-calls` 次 |
| 校验并执行决策 | 训练进程，在 rollout 边界由主线程执行 | 0 |
| 重启、测试把关、记录崩溃 | `supervisor.py` | 0 |
| 改代码、需要拍板的决定 | 你（通过 Remote Control 会话） | 你回复时才调用 |

## 规则（`agent_bridge.Triggers`）

| 规则 | 条件 | 默认值 |
|---|---|---|
| `degraded` | 本次分数低于历史最好的 `degrade_ratio` 倍 | 0.9，比评估回调自带的 0.8 停止线更早 |
| `plateau` | 连续若干次评估都没有出现新的最好成绩 | 4 次 |
| `timeouts` | 超时率超过阈值 | 0.15 |
| `jev_disagrees` | 抽检次数 ≥ 20，且与推演结论的一致率低于阈值 | 0.5 |

没有任何规则触发的评估，**不会调用模型**，只往 `events.jsonl` 写一条 `eval`。
每问一次之后，至少要隔 `--agent-cooldown` 次评估（默认 2 次）才会再问。

## 决策格式：每次只改一个参数

```json
{"action": "none | adjust | stop | escalate_to_human", "param": "<tunable>", "value": 0.004, "reason": "..."}
```

| action | 效果 |
|---|---|
| `none` | 只记录 |
| `adjust` | 把 `param` 限制到合法范围后设置，并写入 `decisions.jsonl`（包含改动前后的值） |
| `stop` | 结束当前训练，保存 `<save>_restart.zip`，写 `runs/restart.json`，以 **退出码 75** 退出 |
| `escalate_to_human` | 写一条 `escalate_to_human` 事件，训练继续跑 |

### 可调的热参数（`train_with_jev.build_tunables`）

| 参数 | 范围 | 说明 |
|---|---|---|
| `ent_coef` | 0–0.02 | 熵奖励 |
| `target_kl` | 0.005–0.05 | KL 超过它的 1.5 倍时提前结束本轮 PPO epoch |
| `lr_scale` | 0.1–3.0 | 学习率 schedule 的倍率（SB3 每次更新都会从 schedule 重新取值，直接改 optimizer 会被覆盖） |
| `clip_scale` | 0.5–2.0 | clip schedule 的倍率 |
| `bc_coef_start` | 0–1 | BC loss 的初始权重 |
| `jev_budget` | 0–128 | 每批标注最多调用 Jev 的次数（加了 `--no-jev` 时没有这一项） |
| `audit_rate` | 0–0.2 | 抽检比例（加了 `--no-jev` 时没有这一项） |

`n_steps`、`batch_size`、`gamma`、`gae_lambda`、奖励函数、网络结构**不在表里**。改这些需要重启，或者属于改代码，由你来决定。

## `runs/` 目录里的文件

| 文件 | 内容 |
|---|---|
| `events.jsonl` | 运行事件：`run_started`、`eval`、`ask`、`ask_skipped`、`adjusted`、`stop_requested`、`escalate_to_human`、`restart_requested`、`run_finished`、`crash`，以及 supervisor 写的 `supervisor_*`、`restarting` |
| `decisions.jsonl` | 每次决策，包含触发原因、改动前后的值、花费。下次提问时会把最近 3 条作为记忆传给 Claude |
| `agent_cost.json` | 累计花费；达到 `--agent-max-usd` 后不再提问 |
| `restart.json` | 退出码 75 时写入：checkpoint 路径、本段已训练的步数、停止原因 |

## 运行

```bash
# 1. 演练：mock 顾问，0 花费，每次评估都提问，用来检查整条链路
.venv/bin/python supervisor.py -- --agent mock --agent-trigger always --no-jev \
    --timesteps 200000 --eval-freq 50000 --eval-episodes 10

# 2. 正式：claude -p（默认用 Haiku），整个运行最多问 10 次、花 2 美元
.venv/bin/python supervisor.py -- --agent claude --jev-budget 32 --save pth/stable_7_jev
```

- `--` 后面的参数原样传给 `train_with_jev.py`。重启时，supervisor 会替换其中的 `--init`、`--timesteps`（剩余步数）和 `--save`（加 `_r1`、`_r2`……，避免覆盖上一段的 `_best`）。
- 重启前会先跑 `python -m unittest discover -q -s tests`（标准库，不需要 pytest；可以用 `--test-cmd` 改）。**测试没过就不重启**，写一条 `escalate_to_human` 后退出。supervisor 不会动 git，也不会改你的源码。
- 已知限制：重启后 LR 和 clip 的 schedule 会从头开始，在剩余步数上重新衰减。

### 需要满足的条件

- 本机已安装 Claude Code 并登录（用订阅额度；`--bare` 模式才需要 API key）。
- `--permission-prompts` 需要 Claude Code v2.1.259 或以上。版本太旧时调用会失败，失败会被记成 `none`，**不影响训练**。

## 人工控制：`train_ctl.py`

所有人工操作都通过它完成，Remote Control 会话里的 Haiku 也只用它：

| 命令 | 作用 |
|---|---|
| `train_ctl.py start -- <训练参数>` | 以独立进程组启动 `supervisor.py`（关掉 Claude 会话后训练照样跑），日志写到 `runs/supervisor.log` |
| `train_ctl.py status` | 用十来行概括：进程是否在跑、最近一次评估、最好成绩、最近 3 条决策、花费、最近的重要事件 |
| `train_ctl.py events -n 15` | 最近的事件，每条压成一行 |
| `train_ctl.py adjust <param> <value> --reason ...` | 先按 `HOT_PARAMS` 校验，再原子写入 `runs/control.json`；训练在下一个 rollout 边界生效 |
| `train_ctl.py stop --reason ...` | 优雅停止：保存 `<save>_final.zip`，退出码 0，supervisor 不会重启 |
| `train_ctl.py kill` | 紧急情况才用：给整个进程组发 SIGTERM，**不保存 checkpoint** |

训练端由 `HumanControlCallback` 读取 `control.json`，这个回调一直开着，不需要加 `--agent`。启动之前就存在的旧命令（seq 不比启动时大）会被忽略。

## 用 Haiku 和 Remote Control 盯训练

项目里已经带了三个 skill 和权限配置：

| 文件 | 作用 |
|---|---|
| `.claude/skills/train-start/SKILL.md` | `/train-start [参数]`：预检 → 你确认命令 → `train_ctl.py start` → 挂上 Monitor。只能手动调用 |
| `.claude/skills/train-watch/SKILL.md` | 收到 Monitor 事件或被问"训练怎么样了"时加载：每种事件怎么处理、何时推送、Monitor 到期后重新挂上 |
| `.claude/skills/train-control/SKILL.md` | 你要求调参或停止时加载：只走 `train_ctl.py`，一次只改一个参数，不在可调参数表里的改动只提方案、不改代码 |
| `.claude/settings.json` | 只放行上面这些命令（`train_ctl.py`、`unittest`、两个固定的 `tail`、`claude --version`），这样在手机上不用一条条批准 |

启动方式：

```bash
cd ~/project/ml/ai_snake
claude --remote-control "ai_snake training" --model haiku
# 在 /config 里打开 "Push when Claude decides" 和 "Push when actions required"
# 然后输入：
/train-start
```

- 用 `--model haiku` 启动整个会话：Monitor 事件触发的每一轮都会跑在 Haiku 上。skill 里的 `model: haiku` 只对加载 skill 的那一轮生效。
- 训练内部的 `claude -p` 顾问默认也是 Haiku（`--agent-model`）。
- 之后在手机上直接说"训练怎么样了"、"把 ent_coef 调到 0.005"、"停掉"就行。
