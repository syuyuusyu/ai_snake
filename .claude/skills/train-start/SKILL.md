---
name: train-start
description: 启动一次由 supervisor 托管、Claude 监督的 ai_snake 训练，并挂上 Monitor 盯关键事件。只在用户输入 /train-start 时使用。
disable-model-invocation: true
model: haiku
---

# 启动受监督的训练

参数：`$ARGUMENTS`（会原样传给 `train_with_jev.py`；为空时用下面的默认值）。

按顺序执行，**任何一步失败就停下来，用中文告诉用户原因，不要自己修代码**。

## 1. 确认当前没有在跑

```bash
.venv/bin/python train_ctl.py status
```
如果输出 `supervisor: running`，告诉用户已经有训练在跑，结束本 skill。

## 2. 预检

```bash
.venv/bin/python -m unittest -q tests.test_agent_bridge tests.test_jev_training tests.test_jev_assist
```
必须以 `OK` 结束。

如果参数里有 `--agent claude`（或者参数为空、要用默认值），再执行：
```bash
claude --version
```
版本必须 ≥ 2.1.259（训练里的 `claude -p` 要用 `--permission-prompts`）。版本太低就提醒用户升级，或者改用 `--agent mock`。

## 3. 确认参数

- `$ARGUMENTS` 为空时，默认参数是：`--agent claude --jev-budget 32 --save pth/stable_7_jev`
- 把最终要执行的完整命令发给用户，**等用户确认后再执行**（正式训练耗时长，而且会花额度）。

## 4. 启动

```bash
.venv/bin/python train_ctl.py start -- <参数>
```
输出里应该有 `started supervisor pid ...`。

## 5. 挂上 Monitor

用 Monitor 工具，参数如下：
- command：
  ```bash
  tail -n 0 -F runs/events.jsonl | grep --line-buffered -E '"kind": "(escalate_to_human|crash|restart_requested|restarting|run_finished|supervisor_finished|control_rejected)"'
  ```
- description：`ai_snake 训练关键事件`
- timeout_ms：`1800000`

## 6. 汇报

用一两句话告诉用户：已经启动、supervisor 的 pid、Monitor 已挂上。之后的事件按 `train-watch` skill 处理。
