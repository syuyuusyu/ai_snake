# stable_4 + gating study (seeds 0–49, 12×12, deterministic policy)

Same seeds/env construction as `evaluate_jev.py` (new `SnakeEnv(seed=s)` per episode).

| Setup | Wins | Mean max len | Median | Outcomes | Advisor calls |
|---|---:|---:|---:|---|---:|
| Jev 1.13 alone (earlier run) | 0/50 | 52.6 | 53 | 50 death | every multi-choice step |
| stable_4 raw | 2/50 | 86.5 | 109.5 | 41 death, 7 timeout | 0 |
| stable_4 + lethal-move mask | 13/50 | 101.6 | 129 | 29 death, 8 timeout | 0 |
| stable_4 + mask + rollout oracle at gates | 25/50 | 128.4 | 143.5 | 15 death, 10 timeout | 6.7 / episode (0.28% of steps) |

- Raw stable_4: 29 of 41 deaths had a safe move available (mostly only one); the policy put ~100% on the lethal move.
- 7 raw timeouts are early-game loops at length 4–43 (deterministic argmax cycling).
- Gates (`jev_assist.detect_gate`): `risk` = PPO move breaks head→tail reachability while another safe move keeps it (306 hits), `loop` = repeated (snake, food) state (29 hits).
- The "oracle" is an upper-bound stand-in for any advisor: 300-step greedy PPO rollouts from each safe move. 49/338 gate decisions were decisive by survival split; the rest were chosen by the soft score (survival + 50×food).
- Remaining failures: 4 early loops (len 4–43), 6 endgame timeouts (128–141), 15 endgame deaths (127–143).

Reproduce: `python simulate_gates.py {safe|tail|oracle} 50 out.jsonl 0`.
`tail` mode (never break tail reachability, rule only) turns deaths into tail-chasing timeouts — a rule alone is not enough at gates.
