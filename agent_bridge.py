"""Let an LLM advisor (Claude via `claude -p`) tune a running training job.

Layering (cheap first):
  1. rules in `should_ask`   decide whether an evaluation is worth a question  -> 0 calls
  2. `advisor(event)`        returns ONE structured decision (read-only, schema-bound)
  3. `AgentBridgeCallback`   validates and applies it on the training thread
  4. supervisor.py           handles restarts (exit code 75), tests, crashes

The callback knows nothing about Claude: any callable `advisor(event) -> (decision, cost_usd)`
works, which is how tests use `mock_advisor`.

Files written under `run_dir` (default `runs/`):
  events.jsonl     one line per notable event; watch it with Monitor / tail -F
  decisions.jsonl  every advisor decision with before/after values (audit log + memory)
  agent_cost.json  accumulated advisor spend in USD
  restart.json     written by train_with_jev.py before exiting with code 75
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from stable_baselines3.common.callbacks import BaseCallback

RESTART_EXIT_CODE = 75
ACTIONS = ["none", "adjust", "stop", "escalate_to_human"]

# Hot parameters: name -> (min, max, integer, description). Shared by train_with_jev.build_tunables
# (which binds them to the live model) and train_ctl.py (which validates human commands).
HOT_PARAMS = {
    "ent_coef": (0.0, 0.02, False, "entropy bonus"),
    "target_kl": (0.005, 0.05, False, "early-stop PPO epochs above 1.5x this KL"),
    "lr_scale": (0.1, 3.0, False, "multiplier on the LR schedule"),
    "clip_scale": (0.5, 2.0, False, "multiplier on the clip schedule"),
    "bc_coef_start": (0.0, 1.0, False, "BC loss weight at progress=1"),
    "jev_budget": (0, 128, True, "max Jev requests per label batch (not with --no-jev)"),
    "audit_rate": (0.0, 0.2, False, "share of decisive states audited by Jev (not with --no-jev)"),
}


# --------------------------------------------------------------------------- #
# Tunable parameters
# --------------------------------------------------------------------------- #
@dataclass
class Tunable:
    """A hot parameter the advisor may change between PPO updates."""
    name: str
    lo: float
    hi: float
    get: Callable[[], float]
    set: Callable[[float], None]
    doc: str = ""
    integer: bool = False

    def clamp(self, value: float) -> float:
        value = min(max(float(value), self.lo), self.hi)
        return int(round(value)) if self.integer else value


class ScaledSchedule:
    """Wrap an SB3 schedule with a mutable multiplier.

    SB3 recomputes learning rate and clip range from their schedules at the start of
    every `train()` call, so editing optimizer param groups directly is overwritten.
    """

    def __init__(self, base: Callable[[float], float], scale: float = 1.0):
        self.base, self.scale = base, scale

    def __call__(self, progress_remaining: float) -> float:
        return self.base(progress_remaining) * self.scale


def build_schema(tunables: dict[str, Tunable]) -> dict:
    """JSON schema for one decision. `param`/`value` allow exactly one change per decision."""
    return {
        "type": "object",
        "properties": {
            "action": {"enum": ACTIONS},
            "param": {"enum": sorted(tunables)},
            "value": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["action", "reason"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# Run directory: events, decisions, cost
# --------------------------------------------------------------------------- #
class RunLog:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _append(self, name: str, record: dict):
        with self.lock, (self.dir / name).open("a") as fh:
            fh.write(json.dumps(record, default=float) + "\n")

    def event(self, kind: str, **data):
        self._append("events.jsonl", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **data})

    def decision(self, record: dict):
        self._append("decisions.jsonl", record)

    def recent_decisions(self, k: int = 3) -> list[dict]:
        path = self.dir / "decisions.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()[-k:] if line.strip()]

    def add_cost(self, usd: float) -> float:
        path = self.dir / "agent_cost.json"
        with self.lock:
            total = json.loads(path.read_text())["total_usd"] if path.exists() else 0.0
            total += float(usd or 0.0)
            path.write_text(json.dumps({"total_usd": total}) + "\n")
        return total

    def total_cost(self) -> float:
        path = self.dir / "agent_cost.json"
        return json.loads(path.read_text())["total_usd"] if path.exists() else 0.0


# --------------------------------------------------------------------------- #
# Advisors
# --------------------------------------------------------------------------- #
PROMPT = """You supervise a MaskablePPO Snake training run. A rule fired on the latest evaluation.
Event (JSON):
{event}

Context files you may read: jev_training_plan.md, agent_supervision.md, {run_dir}/decisions.jsonl.
Return exactly one decision:
- "none": noise or no clear action.
- "adjust": change ONE tunable (param + value, within its range); prefer small moves.
- "stop": stop and save for a restart (only for problems a hot parameter cannot fix).
- "escalate_to_human": needs a code change or a judgement call; explain in reason.
Evaluations use few seeds and are noisy; earlier decisions are in event.recent_decisions.
"""


def make_claude_advisor(schema: dict, *, model: str = "haiku", max_budget_usd: float = 0.5,
                        claude_bin: str = "claude", cwd: str | Path = ".", run_dir: str | Path = "runs",
                        timeout: int = 600, extra_args: list[str] | None = None):
    """Return advisor(event) -> (decision, cost) backed by `claude -p` (read-only)."""

    def advisor(event: dict):
        cmd = [claude_bin, "-p", PROMPT.format(event=json.dumps(event, indent=1, default=float), run_dir=run_dir),
               "--model", model, "--output-format", "json", "--json-schema", json.dumps(schema),
               "--allowedTools", "Read", "--permission-prompts", "none",
               "--max-budget-usd", str(max_budget_usd), *(extra_args or [])]
        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"action": "none", "reason": f"advisor unavailable: {exc}"}, 0.0
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            out = {}
        cost = float(out.get("total_cost_usd") or 0.0)
        decision = out.get("structured_output")
        if proc.returncode != 0 or not isinstance(decision, dict):
            detail = (out.get("result") or proc.stderr or "")[:300]
            return {"action": "none", "reason": f"advisor failed (rc={proc.returncode}): {detail}"}, cost
        return decision, cost

    return advisor


def mock_advisor(event: dict):
    """Deterministic stand-in for tests and dry runs; costs nothing."""
    m, triggers = event["metrics"], event["triggers"]
    values = event["tunables"]
    if "degraded" in triggers:
        return {"action": "escalate_to_human", "reason": "mock: score degraded"}, 0.0
    if "timeouts" in triggers and "ent_coef" in values:
        return {"action": "adjust", "param": "ent_coef", "value": values["ent_coef"]["value"] * 1.5,
                "reason": f"mock: timeout_rate={m.get('timeout_rate')}"}, 0.0
    return {"action": "none", "reason": "mock: nothing to do"}, 0.0


# --------------------------------------------------------------------------- #
# The callback
# --------------------------------------------------------------------------- #
@dataclass
class Triggers:
    degrade_ratio: float = 0.9      # ask before FixedSeedEvalCallback's own 0.8 stop rule fires
    plateau_evals: int = 4          # evaluations without a new best
    timeout_rate: float = 0.15
    audit_min: int = 20             # audits needed before trusting the agree rate
    audit_agree: float = 0.5
    always: bool = False            # dry-run/testing: ask after every evaluation


class AgentBridgeCallback(BaseCallback):
    def __init__(self, eval_cb, tunables: dict[str, Tunable], advisor, log: RunLog, *,
                 stats_fn: Callable[[], dict] | None = None, triggers: Triggers | None = None,
                 cooldown_evals: int = 2, max_calls: int = 10, max_usd: float = 2.0):
        super().__init__()
        self.eval_cb, self.tunables, self.advisor, self.log = eval_cb, tunables, advisor, log
        self.stats_fn = stats_fn or (lambda: {})
        self.trig = triggers or Triggers()
        self.cooldown_evals, self.max_calls, self.max_usd = cooldown_evals, max_calls, max_usd
        self.inbox: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.calls = 0
        self.seen_evals = 0
        self.cooldown_until = 0
        self.best_seen, self.best_at = float("-inf"), 0
        self.restart_requested = False
        self.restart_reason = ""

    # ---- rules ----------------------------------------------------------- #
    def triggers_for(self, m: dict, n_evals: int, best: float) -> list[str]:
        if self.trig.always:
            return ["always"]
        out = []
        if best > float("-inf") and m["score"] < self.trig.degrade_ratio * best:
            out.append("degraded")
        if n_evals - self.best_at >= self.trig.plateau_evals:
            out.append("plateau")
        if m.get("timeout_rate", 0.0) > self.trig.timeout_rate:
            out.append("timeouts")
        st = self.stats_fn()
        if st.get("audit", 0) >= self.trig.audit_min and \
                st.get("audit_agree", 0) / max(st["audit"], 1) < self.trig.audit_agree:
            out.append("jev_disagrees")
        return out

    def snapshot_tunables(self) -> dict:
        return {k: {"value": t.get(), "min": t.lo, "max": t.hi, "doc": t.doc} for k, t in self.tunables.items()}

    # ---- SB3 hooks ------------------------------------------------------- #
    def _on_step(self) -> bool:
        return not self.restart_requested

    def _on_rollout_end(self) -> None:
        while not self.inbox.empty():
            self.apply(*self.inbox.get())
        n = getattr(self.eval_cb, "n_evals", 0)
        m = getattr(self.eval_cb, "last_metrics", None)
        if m is None or n == self.seen_evals:
            return
        self.seen_evals = n
        best = self.eval_cb.best_score
        if best > self.best_seen:
            self.best_seen, self.best_at = best, n
        triggers = self.triggers_for(m, n, best)
        self.log.event("eval", t=self.num_timesteps, n_evals=n, metrics=m, best=best, triggers=triggers)
        if not triggers or self.restart_requested:
            return
        model = getattr(self, "model", None)   # unset when the callback is used outside learn()
        total = getattr(model, "_total_timesteps", None)
        rollout = getattr(model, "n_steps", 0) * getattr(model, "n_envs", 0)
        ending = total is not None and self.num_timesteps + rollout >= total
        skip = ("ending" if ending   # the answer could not be applied before learn() returns
                else "busy" if self.worker is not None and self.worker.is_alive()
                else "cooldown" if n < self.cooldown_until
                else "max_calls" if self.calls >= self.max_calls
                else "max_usd" if self.log.total_cost() >= self.max_usd
                else None)
        if skip:
            self.log.event("ask_skipped", t=self.num_timesteps, reason=skip, triggers=triggers)
            return
        event = {"t": self.num_timesteps, "n_evals": n, "triggers": triggers, "metrics": m, "best": best,
                 "tunables": self.snapshot_tunables(), "jev": self.stats_fn(),
                 "recent_decisions": self.log.recent_decisions(3)}
        self.calls += 1
        # Every question (not just adjustments) starts a cooldown, and a plateau re-arms only
        # after another full plateau window, so a persistent condition is not asked about twice.
        self.cooldown_until = n + self.cooldown_evals
        if "plateau" in triggers:
            self.best_at = n
        self.log.event("ask", t=self.num_timesteps, triggers=triggers, call=self.calls)
        self.worker = threading.Thread(target=self._ask, args=(event,), daemon=True)
        self.worker.start()

    def _ask(self, event):
        try:
            decision, cost = self.advisor(event)
        except Exception as exc:  # advisor failures never kill training
            decision, cost = {"action": "none", "reason": f"advisor error: {exc!r}"}, 0.0
        self.log.add_cost(cost)
        self.inbox.put((event, decision, cost))

    def _on_training_end(self) -> None:
        if self.worker is not None:
            self.worker.join(timeout=5)
        while not self.inbox.empty():
            self.apply(*self.inbox.get())

    # ---- apply on the training thread ------------------------------------ #
    def apply(self, event: dict, decision: Any, cost: float = 0.0) -> dict:
        record = {"t": self.num_timesteps, "asked_at": event.get("t"), "triggers": event.get("triggers"),
                  "decision": decision, "cost_usd": cost, "applied": False}
        action = decision.get("action") if isinstance(decision, dict) else None
        if action == "adjust":
            tun = self.tunables.get(decision.get("param"))
            value = decision.get("value")
            if tun is None or not isinstance(value, (int, float)):
                record["error"] = "unknown param or non-numeric value"
            else:
                before, after = tun.get(), tun.clamp(value)
                tun.set(after)
                record.update(applied=True, param=tun.name, before=before, after=after)
                self.cooldown_until = self.seen_evals + self.cooldown_evals
                self.log.event("adjusted", t=self.num_timesteps, param=tun.name, before=before, after=after,
                               reason=decision.get("reason"))
        elif action == "stop":
            self.restart_requested, self.restart_reason = True, decision.get("reason", "")
            record["applied"] = True
            self.log.event("stop_requested", t=self.num_timesteps, reason=self.restart_reason)
        elif action == "escalate_to_human":
            record["applied"] = True
            self.log.event("escalate_to_human", t=self.num_timesteps, reason=decision.get("reason"),
                           metrics=event.get("metrics"))
        elif action != "none":
            record["error"] = f"invalid action {action!r}"
        self.log.decision(record)
        return record


# --------------------------------------------------------------------------- #
# Human control: runs/control.json (written by train_ctl.py)
# --------------------------------------------------------------------------- #
class HumanControlCallback(BaseCallback):
    """Apply commands from `<run_dir>/control.json` at rollout boundaries.

    {"seq": 3, "action": "adjust", "param": "ent_coef", "value": 0.005, "reason": "..."}
    {"seq": 4, "action": "stop", "reason": "..."}   -> save <save>_final and exit 0 (no restart)

    Commands whose seq is not larger than the one present at start-up are ignored, so a stale
    file from an earlier run never fires.
    """

    def __init__(self, tunables: dict[str, Tunable], log: RunLog):
        super().__init__()
        self.tunables, self.log = tunables, log
        self.path = log.dir / "control.json"
        self.seq = self._read().get("seq", 0)
        self.stop_requested, self.stop_reason = False, ""

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _on_step(self) -> bool:
        return not self.stop_requested

    def _on_rollout_end(self) -> None:
        cmd = self._read()
        seq = cmd.get("seq", 0)
        if not isinstance(seq, int) or seq <= self.seq:
            return
        self.seq = seq
        record = {"t": self.num_timesteps, "source": "human", "seq": seq, "decision": cmd, "applied": False}
        if cmd.get("action") == "adjust" and cmd.get("param") in self.tunables \
                and isinstance(cmd.get("value"), (int, float)):
            tun = self.tunables[cmd["param"]]
            before, after = tun.get(), tun.clamp(cmd["value"])
            tun.set(after)
            record.update(applied=True, param=tun.name, before=before, after=after)
            self.log.event("adjusted", t=self.num_timesteps, source="human", param=tun.name,
                           before=before, after=after, reason=cmd.get("reason"))
        elif cmd.get("action") == "stop":
            self.stop_requested, self.stop_reason = True, cmd.get("reason", "")
            record["applied"] = True
            self.log.event("stop_requested", t=self.num_timesteps, source="human", reason=self.stop_reason)
        else:
            record["error"] = "invalid command or param not available in this run"
            self.log.event("control_rejected", t=self.num_timesteps, seq=seq, command=cmd)
        self.log.decision(record)
