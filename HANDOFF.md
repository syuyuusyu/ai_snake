# 当前状态与操作说明（2026-10-11）

给接手这个项目的人看。拉取代码后，先读完这份文件再开始训练。

## 1. 一句话结论

**目前最好的方案是：开着 `Shield` 从头训练的模型 `pth/stable_shield_1.zip`，配合 `Shield` 使用。**
- 种子 2000–2399（400 局，挑检查点时没用过）：胜率 **99.2%**，超时 0.8%，死亡 0。
- 同样条件下 `stable_4` + `Shield` 的胜率是 81%。

**注意：这个模型离开 Shield 完全不能用**，不开 Shield 时胜率是 0（见第 6 节）。游戏和评估默认都开着 Shield，所以正常使用没问题。

在 `stable_4` 上继续训练会越练越差。原因很可能是奖励规则变了，而价值网络和策略共用同一个卷积网络（见第 7 节）。

## 2. Git 状态

- 分支：`master`。
- `pth/` 里只保留两个模型：
  - `stable_shield_1.zip`：第 5.2 节 B 的 15M 步检查点（原 `pth/exp/scr_14999040_steps.zip`），当前最好的模型；
  - `stable_4.zip`：基线，也是 `train_with_jev.py --init` 的默认值。
- 其他旧模型、`pth/exp/` 的检查点和 Jev 评估的原始数据（`evaluations/jev_*`）已删除。完整备份在 Windows 机器上：`C:\code\project\ml\ai_snake-backup-20261011.bundle`（`git clone <bundle>` 即可还原）。
- 不进 git 的：`pth/exp/`（实验检查点，要留的复制成 `pth/<名字>.zip` 再提交）、`runs/`、`logs/`、`.vscode/`。

### 2.1 转到 Mac 后要注意的地方

- **重新 clone，不要在旧目录里 pull。**为了缩小仓库，git 历史可能已被重写（强制推送）。如果 Mac 上已有旧 clone，`git pull` 会把旧历史合并回来，旧的大文件又会被推上去。做法：
  ```
  mv ai_snake ai_snake_old        # 先备份旧目录，里面的 logs/ 等本地文件需要的话手动拷过来
  git clone git@github.com:syuyuusyu/ai_snake.git
  ```
- **环境**：`uv sync` 在 Mac 上会从 PyPI 装 torch（用 MPS）并装 `mlx`。第 3 节关于 cu124 和南大镜像的内容只针对 Windows，Mac 上不用管。
- **训练速度和参数**：第 5.2 节的速度（约 19k 步/秒）和 `--workers 22` 是 RTX 4080 + 多核 Windows 机器上的数据。Mac 上 `--workers` 默认取 `min(16, CPU 核数)`，不要照抄 22；先跑一小段确认速度，再估算 15M 步要多久。
- **`train_ctl.py` 和 `.claude/skills/` 在 Mac 上可以用**：第 9 节的问题只出现在 Windows 上。
- **`play_with_model.py` 默认模型**已改为 `pth/stable_shield_1.zip`（原来是已删除的 `stable_5.zip`）。
- **MLX 权重**：`pth/snake_policy.safetensors` 已删除，仓库里没有代码读取它。需要时在 Mac 上重新生成：
  ```
  uv run python convert_to_mlx.py --model pth/stable_4.zip --out pth/snake_policy.safetensors
  ```
  原来的文件就是这样从 `stable_4` 转出来的。`stable_shield_1` 的网络结构可能不同（文件更大），还没试过能不能转。
- **Windows 专用的内容**：`safe_env.py` 不导入 torch 是为了避开 Windows 的内存提交上限，Mac 上没有这个问题，但保持现状即可，不用改回去。

## 3. 安装环境（uv）

项目用 uv 管理依赖（`pyproject.toml` + `uv.lock`），Python 3.11：
```
uv sync
```
- `mlx` 只在 macOS 上安装（`sys_platform == 'darwin'`）。
- **torch**：Windows/Linux 上从 PyTorch 的 cu124 源安装 CUDA 版；Mac 上仍从 PyPI 安装，用 MPS。
- **国内下载 torch 很慢时**：可以先从南大镜像（`mirrors.nju.edu.cn/pytorch/whl/cu124/`）下好 `torch-2.4.1+cu124-cp311-cp311-win_amd64.whl`，再用 `uv pip install <whl>` 装上。
  - 装好后要删掉 `.venv/Lib/site-packages/torch-2.4.1+cu124.dist-info/direct_url.json`，以及 `RECORD` 里对应的那一行，否则 `uv sync` 会认为来源不同，重新下载 2.5GB。
  - 检查办法：`uv sync --dry-run` 显示 "Would make no changes" 就对了。
- 检查显卡：
  ```
  uv run python -c "import torch; print(torch.cuda.is_available())"
  ```
- 跑测试：
  ```
  uv run python -m unittest tests.test_agent_bridge tests.test_board_dimensions tests.test_evaluate_jev tests.test_game_env tests.test_jev_training tests.test_play_with_jev tests.test_worker_vec_env
  uv run python -m pytest tests/test_jev_assist.py
  ```
  Windows 上 `test_agent_bridge` 有 2 个用例本来就会失败（WinError 193：测试用的假可执行文件是 shell 脚本），和本次改动无关。

## 4. 常用操作

**评估模型**：
```
uv run python evaluate_model.py pth/stable_shield_1.zip
uv run python evaluate_model.py pth/stable_4.zip --no-shield
```
- 默认用种子 1000–1199 共 200 局。训练过程中的评估用的是种子 0–49，所以比较模型时要用另一组种子。
- 这批检查点是看 1000–1199 的结果挑出来的，最终确认用的是 2000–2399。

**看蛇在屏幕上玩**：
```
uv run python play_with_model.py --model pth/stable_shield_1.zip
```

## 5. 实验和结论

### 5.1 Shield 的作用（`stable_4`，种子 1000–1199）

| 设置 | 胜率 | 超时 | 死亡 | 得分 |
|---|---|---|---|---|
| 只屏蔽立即会死的走法 | 18% | 17% | 65% | 119 |
| + 不自困、不绕圈 | 68% | 32% | 0% | 230 |
| + 饿了就走安全最短路去吃（当前 Shield） | 79% | 21% | 0% | 255 |
| 对照：随机走 + Shield | 4% | – | – | 140 |

得分 = 最大长度，赢了再加 144（棋盘面积）。

### 5.2 继续训练 vs 从头训练（2026-10-11，RTX 4080）

| 实验 | 种子 1000–1199 胜率 | 种子 2000–2399 胜率 |
|---|---|---|
| `stable_4`（基线） | 79% | 81.0% |
| A：从 `stable_4` 继续训练，学习率 3e-6→1e-6，3M 步 | 0.5M 步 78% → 1.5M 步 70.5% → 3M 步 **59.5%** | – |
| A′：A + `--timeout-terminal` | 77.5% → 69.5% → **68.0%** | – |
| **B：从头训练 + `--timeout-terminal`，15M 步** | **100%**（满分 288） | **99.2%** |
| B，25M 步 | 93.5%（取最大概率动作）/ 100%（按概率采样） | – |

- 以上都开着 Shield，默认取最大概率的动作。
- B 的训练过程中，滚动对局的胜率从 5–10M 步的 98% 上升到 20–25M 步的 100%。
- 25M 步在取最大概率动作时反而变差，所以最终用 **15M 步的检查点**。
- B 用时：15M 步约 13 分钟，25M 步约 22 分钟，速度约 19k 步/秒。

训练命令：
```
# B：从头训练
uv run python train_with_jev.py --init none --no-jev --no-labels --timeout-terminal --envs 256 --n-steps 128 --workers 22 --timesteps 25000000 --eval-freq 0 --checkpoint-freq 2500000 --seed 0 --save pth/exp/scr --log-dir logs/exp_scr
# A′：继续训练（A 去掉 --timeout-terminal）
uv run python train_with_jev.py --no-jev --no-labels --timeout-terminal --envs 64 --n-steps 512 --workers 22 --timesteps 3000000 --eval-freq 0 --checkpoint-freq 500000 --seed 0 --save pth/exp/ftT --log-dir logs/exp_ftT
```

**尚未验证的地方：**
- 只跑了一个训练种子（seed 0）。
- 没有"从头训练但不加 `--timeout-terminal`"的对照，所以分不清 99% 里从头训练和超时修正各占多少功劳。

### 5.3 超时被截断的 bug（`--timeout-terminal`）

- 环境把"576 步没吃到食物"标成 `truncated`。SB3 会把它当作时间限制，补回 `γ·V(最后状态)`（`sb3_contrib` 的 `ppo_mask.py:253-260`）。
- 蛇很长时这个值很大，−1 的超时惩罚就被抵消了，模型学不到"兜圈超时是坏事"。
- `--timeout-terminal`（`safe_env.TimeoutIsTerminal`）在训练时把超时当作真正的结束。只影响训练，不影响评估。

### 5.4 Jev / BC 没有帮助（Mac 上的结论）

- 策略和 Jev 标签的一致率一直只有 44%–51%。
- 标签只攒了几百条却反复抽样训练，BC 只是让策略更不稳定。
- 建议不要再开 Jev（每次调用都要花钱）。

## 6. 新模型离不开 Shield

- 不开 Shield 时（只屏蔽立即会死的走法），`scr_15M` 胜率 0%：蛇长到 6 左右就开始原地兜圈，最后超时。
- 对比：`stable_4` 训练时没有这个 Shield，单独上场还有 18%。

在 60 局里统计了 Shield 改掉模型原本选择（取最大概率的动作）的情况：

| 蛇长 | scr_15M：每局步数 / Shield 改动次数 | stable_4：每局步数 / Shield 改动次数 |
|---|---|---|
| < 12 | 574 / ~39 | 126 / ~4 |
| 12–59 | 4471 / ~192 | 879 / ~6 |
| 60+ | 2291 / ~23 | 2192 / ~25 |

- 总共只有 3.5% 的步数被改动，但都是关键的几步：会形成死循环或把自己困住的那几步。
- 新模型在前中期吃一个食物要花 stable_4 四到五倍的步数。它学会的打法是"随便逛，兜圈时 Shield 帮我打破，饿了 Shield 带我去吃"。

**为什么继续训练不会让模型"学会"Shield：**
1. Shield 是训练环境的一部分。它把模型的错误都挡掉了，错误没有任何代价，模型也就没有理由去改。
2. MaskablePPO 把被屏蔽动作的输出设为 −1e8，这些位置梯度为 0，网络从没被告诉过"这一步不好"。
3. 网络只看当前一帧画面（`snake_game.get_obs`），看不到历史：
   - "不绕圈"需要记住之前的局面；
   - "饿了"需要计数没吃到食物的步数。

   这两条规则网络都学不到。

如果需要一个离开 Shield 也能玩的模型，思路有三个：
- **惩罚 Shield 的干预**：模型照自己的选择出手，被 Shield 改动就扣分。
- **蒸馏**：加一个监督损失，让网络不加屏蔽时的输出，也给 Shield 禁掉的动作很低的概率。
- **逐步撤掉 Shield**：一条一条去掉规则，或者让 Shield 生效的概率逐渐降低。

## 7. 为什么在 `stable_4` 上继续训练会退化

`stable_4` 是 2025-03 按旧奖励训练的（`45bf479`），和现在的环境差别很大：

| | stable_4 训练时 | 现在的环境（`6d51282` 起） |
|---|---|---|
| 吃食物 | 长度/144 | +1 |
| 胜利 | +100 | +30 |
| 死亡 | −0.1×141^((144−长度)/141)，短蛇约 −14 | −(1 + 4×进度²) |
| 饿死 | 144 步不吃，算死亡 | 576 步，−1（被截断，见 5.3） |
| 其他 | 蛇长 ≤ 30 时：靠近食物 +1/长度，远离 −1/长度；蛇长 > 72 时每步按可达空间加分 | 每步 −0.002，外加安全势能项 |
| 折扣 γ | 0.98 | 0.995 |

- 继续训练时，奖励、折扣、超时规则和 Shield 同时变了，价值网络的预测全部失准。
- 已核对：`stable_4` 的策略和价值网络共用同一个卷积网络（两边权重完全相同）。价值网络误差很大，它的梯度会去改共用的特征，所以学习率再小，策略也会被拖坏。
- 这是推断，还没有单独做实验验证。

在 `stable_4` 上继续训练前，应先冻结策略，只训练价值网络一段时间（最好让价值网络单独用一份卷积网络），然后再联合训练。

**现在的奖励里已经有 Shield 规则的"奖励版"：**
- "头能到尾"：安全势能 Φ = 0.7×可达空间比例 + 0.3×头能到尾，奖励 0.5×(γΦ′ − Φ)；
- "饿了"：超时 −1，每步 −0.002。

但它们远没有 Shield 那样强：
- 让蛇头够不到尾巴只扣 0.15 分，吃到食物却有 +1，模型冒这个险是划算的。
- 这种势能形式的奖励在理论上不改变最优策略，只是提前给提示。

## 8. 代码改动（2026-10-11）

- **`pyproject.toml` / `uv.lock`**：`mlx` 只在 macOS 上安装；Windows/Linux 用 CUDA 版 torch。
- **`safe_env.py`（新）**：放 `SafeSnakeEnv`、`JevGateWrapper`、`TimeoutIsTerminal`、`MaskedMonitor` 和子进程循环。这个文件**不能导入 torch**：Windows 上每个导入 CUDA torch 的进程会占约 1.3GB 内存提交量，64 个进程就超出上限（WinError 1455）。
- **`worker_vec_env.py`（新）**：`WorkerVecEnv` 让一个子进程同时跑多个环境，并把动作屏蔽随每步结果一起返回。
  - 速度从约 12k 步/秒提到 18–20k 步/秒。
  - `tests/test_worker_vec_env.py` 验证了它和 `DummyVecEnv` 的逐步结果完全一致。
- **`jev_assist.py`**：`torch` 改为在用到的函数里才导入。
- **`train_with_jev.py`**：新增以下选项，原有默认值不变：
  - `--init none`：从头训练，默认超参见 `SCRATCH`；
  - `--lr a:b`、`--clip a:b`、`--ent-coef`、`--n-epochs`、`--target-kl`：调整训练参数；
  - `--timeout-terminal`：超时修正；
  - `--no-labels`：不做 Jev 标注，只跑纯 PPO；
  - `--workers`：子进程数；
  - `--checkpoint-freq`：定期保存检查点；
  - 另外关闭了 torch 分布的参数检查，rollout 快约 15%。

## 9. 在 Windows 上训练的注意事项

**不要用 `train_ctl.py`**：
- 它检查训练是否还在跑时调用了 `os.kill(pid, 0)`，在 Windows 上会直接结束那个进程；
- `stop`、`kill` 用的是 `os.killpg`，Windows 上没有。

`.claude/skills/` 里的三个 skill 依赖 `train_ctl.py`，Windows 上也不要用。直接运行第 5.2 节的训练命令即可。

## 10. 可以接着做的方向

1. **复现**：换 1–2 个训练种子，按 B 的设置各跑 15M 步（每次约 13 分钟），确认 99% 能复现。
2. **对照实验**：从头训练但不加 `--timeout-terminal`，分清两者各自的贡献。
3. **离开 Shield 也能玩的模型**（如果需要）：
   - (a) 不开 Shield 从头训练（只屏蔽立即会死的走法），加超时修正；
   - (b) 在 (a) 的基础上，加重"头够不到尾"的一次性惩罚；
   - (c) `stable_4` 先单独训练价值网络，再按 (b) 的奖励继续训练。

   预期单独上场的胜率会明显好于 18%，但很难到 99%。需要先给 `train_with_jev.py` 加"不开 Shield"和奖励强度的选项。
4. **残局**：新模型的超时已降到 0.8%。如果还要消除，可以在残局切换成沿哈密顿回路走。
