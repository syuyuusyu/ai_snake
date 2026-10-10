# 当前状态与操作说明（2026-10-10）

给在另一台电脑（Windows）上接手这个项目的人看。拉取代码后，先读完这份文件再开始训练。

## 1. 一句话结论

**目前最好的方案是 `pth/stable_4.zip` 加上 `Shield` 动作规则。** 在 200 局没用来调参的种子上：胜率 79%，死亡率 0。
在 `stable_4` 的基础上继续做 PPO 微调已经验证过没有提升，**直接用现在的命令重跑训练，算力再强也不会更好**（见第 5 节）。

## 2. Git 状态

- 分支：`master`。
- 本地没有提交的东西（不影响使用）：
  - `pth/stable_7_*`、`stable_8_*`、`stable_9_*`、`test_run_*` 等实验检查点。都没有超过 `stable_4.zip`，所以没提交。`pth/stable_4.zip` 已经在仓库里。
  - `runs/`（训练日志）、`evaluations/*_cache.jsonl`（Jev 标注缓存）、`.claude/settings.local.json`。
- 如果 Windows 上也有未提交的改动：先 `git stash`（或先提交），再 `git pull origin master`，然后处理冲突。

## 3. 在 Windows 上安装环境

1. Python 3.11，在项目根目录建虚拟环境：
   ```
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```
2. **`requirements.txt` 里的 `mlx` 只支持苹果芯片的 Mac，Windows 上装不了**，安装时要跳过：
   ```
   findstr /v /b "mlx" requirements.txt > req-win.txt
   pip install -r req-win.txt
   ```
3. 要用显卡训练，需要按 https://pytorch.org 上的说明重新安装带 CUDA 的 torch，然后检查：
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   打印 `True` 时，训练会自动用显卡（`train_with_ppo.py` 里的 `device` 会依次选 cuda → mps → cpu）。
4. 跑测试，确认环境没问题：
   ```
   python -m unittest tests.test_agent_bridge tests.test_board_dimensions tests.test_evaluate_jev tests.test_game_env tests.test_jev_training tests.test_play_with_jev
   python -m pytest tests/test_jev_assist.py
   ```
   `tests/test_jev_assist.py` 是 pytest 风格的测试，要先 `pip install pytest`。前一行应该显示 `OK`（74 个测试）。

## 4. 常用操作

**评估模型**（推荐先跑一次，确认结果和下表一致）：
```
python evaluate_model.py pth/stable_4.zip
python evaluate_model.py pth/stable_4.zip --no-shield
```
默认用种子 1000–1199 共 200 局。训练过程中的评估用的是种子 0–49，所以比较模型时要用另一组种子，结果才公平。
200 局评估在 Mac 上要几分钟；加上 Shield 后对局更长，所以会更慢。

**看蛇在屏幕上玩**（默认模型是 `stable_5`，要手动指定）：
```
python play_with_model.py --model pth/stable_4.zip
```

## 5. 做过的实验和结论

所有结果都是在种子 1000–1199（200 局）上测出来的。

| 设置 | 胜率 | 超时 | 死亡 | 得分 |
|---|---|---|---|---|
| `stable_4`，只屏蔽立即会死的走法 | 18% | 17% | 65% | 119 |
| + 不自困、不绕圈 | 68% | 32% | 0% | 230 |
| **+ 饿了就走安全最短路去吃（当前默认）** | **79%** | 21% | 0% | 255 |

得分 = 最大长度，赢了再加 144（棋盘面积）。

**PPO 微调没有提升：**
- 从 `stable_4` 出发，学习率 3e-6 → 1e-6，训练 200 万步：分数在约 50 万步后开始下降，训练奖励也一起下降，最后因为退化被自动停止。
- 关闭 BC 后还是会退化，只是晚一些。
- 改成 60 万步（让学习率在训练内完整衰减），3 个种子：不加规则时都比起点低；加上规则后都和起点持平（222–229 vs 230）。

**Jev / BC 没有帮助：**
- 训练时策略和 Jev 标签的一致率一直只有 44%–51%，训练过程中没有提高。
- 标签只攒了 300–800 条，却每轮抽 4×256 条反复训练。BC 只是让策略更不稳定。
- 建议不要再开 Jev（每次调用都要花钱）。

## 6. Shield 规则（`jev_assist.py`）

在原来"不撞墙、不撞身体"的基础上，按顺序缩小可选的走法。某条规则下一步都走不了时，就忽略这条规则：

1. **不自困**：不走会让蛇头够不着蛇尾的那一步（`head_reaches_tail`）。
2. **不绕圈**：上次吃到食物后，不再回到已经出现过的局面（同样的蛇身 + 食物）。
3. **饿了就去吃**：连续 2 × 棋盘面积（12×12 棋盘是 288）步没吃到食物，就算出到食物的最短路，**并且**确认吃到之后蛇头仍然能到达蛇尾，确认通过就强制走这条路（`safe_food_move`）。

除此之外，走法仍然由神经网络决定。规则在训练（`train_with_jev.py` 的 `SafeSnakeEnv`）和屏幕游戏（`snake_game.py`）里都默认开启。
阈值 288 是用种子 0–49 调出来的：144–360 之间效果差不多，太小（0、36）反而更差。

**剩下 21% 都是残局超时**：蛇已经长到 143/144，只差最后一格，但棋盘几乎被身体填满，蛇只能沿一个固定的圈走，正好绕不到最后那格食物。试过两种更精细的路径检查，都没有改善。要解决这个问题，需要在残局切换成"沿着覆盖整个棋盘的固定路线走"（哈密顿回路）。

## 7. 在 Windows 上训练的注意事项

**不要用 `train_ctl.py`。** 它检查训练是否还在跑时调用了 `os.kill(pid, 0)`：在 macOS/Linux 上这只是检查，但**在 Windows 上会直接结束那个进程**。另外 `stop`、`kill` 用的是 `os.killpg`，Windows 上没有。`.claude/skills/` 里的三个 skill 都依赖 `train_ctl.py`，并且写的是 `.venv/bin/python`（Windows 上是 `.venv\Scripts\python`），所以在 Windows 上也不要用这三个 skill。

直接运行训练脚本（参数和 Mac 上最后一次实验一致）：
```
python train_with_jev.py --agent off --no-jev --bc-min-size 100000000 --timesteps 600000 --eval-freq 100000 --seed 0 --save pth/win_s0
```
- `--no-jev --bc-min-size 100000000`：关闭 Jev，也关闭 BC。
- 默认从 `pth/stable_4.zip` 开始（`--init`），64 个并行环境（`--envs`）。
- 训练结束后会生成 `pth/win_s0_best.zip` 和 `pth/win_s0_final.zip`，再用 `evaluate_model.py` 和 `stable_4.zip` 比较。

**但再次强调**：这套设置在 Mac 上已经验证过不会超过 `stable_4`。现在的训练只是"从 `stable_4` 出发、学习率很小的轻微调整"。想让更强的算力真正派上用场，需要换成"从头训练、更大的学习率、几千万到上亿步"的路线。这需要先改训练代码（目前学习率、初始模型都写死在 `train_with_jev.py` 里），还没有做。

## 8. 可以接着做的方向

按"投入小、把握大"排序：

1. **残局规则**：蛇很长时，改成沿哈密顿回路走，目标是消除剩下 21% 的超时。
2. **改超时规则或奖励**：现在超时上限是 4 × 棋盘面积 = 576 步，罚分 -1，和死亡差不多。蛇很长时可以放宽步数限制。改奖励就需要重新训练。
3. **从头训练**：用更强的电脑，配合更大的学习率、课程学习（先在小棋盘练）和更长的训练。时间成本最高，结果也最不确定。
