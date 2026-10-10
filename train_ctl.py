"""Small, validated control CLI for a supervised training run (used by the .claude/skills).

    python train_ctl.py start -- --agent claude --jev-budget 32 --save pth/stable_7_jev
    python train_ctl.py status
    python train_ctl.py events -n 15
    python train_ctl.py adjust ent_coef 0.005 --reason "timeouts stay at 30%"
    python train_ctl.py stop --reason "enough for today"      # graceful: saves <save>_final
    python train_ctl.py kill                                  # emergency only: no checkpoint

Everything goes through <run-dir> files (default runs/), so this works from any shell,
including a Remote Control session on a phone. Output is short on purpose.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from agent_bridge import HOT_PARAMS

NOTIFY = {"escalate_to_human", "crash", "restart_requested", "restarting", "run_finished",
          "supervisor_finished", "stop_requested", "control_rejected", "adjusted"}


def read_jsonl(path: Path, n: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [json.loads(line) for line in (lines[-n:] if n else lines)]


def supervisor_pid(run_dir: Path) -> int | None:
    try:
        pid = int((run_dir / "supervisor.pid").read_text())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def fmt_event(e: dict) -> str:
    m = e.get("metrics") or {}
    bits = [e.get("time", "")[11:19], e["kind"]]
    if "t" in e:
        bits.append(f"t={e['t']}")
    if "score" in m:
        bits.append(f"score={m['score']:.1f} len={m.get('mean_max_length', 0):.1f} "
                    f"win={m.get('victory_rate', 0):.0%} timeout={m.get('timeout_rate', 0):.0%}")
    for key in ("triggers", "param", "before", "after", "reason", "rc", "checkpoint"):
        if e.get(key) not in (None, "", []):
            bits.append(f"{key}={e[key]}")
    return " ".join(str(b) for b in bits)


def cmd_start(args) -> int:
    run_dir = Path(args.run_dir)
    if (pid := supervisor_pid(run_dir)):
        print(f"already running (supervisor pid {pid}); use status/stop")
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    if "--run-dir" not in train_args and args.run_dir != "runs":
        train_args += ["--run-dir", args.run_dir]
    log = (run_dir / "supervisor.log").open("a")
    proc = subprocess.Popen([sys.executable, "supervisor.py", *args.sup_args, "--", *train_args],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(2)
    if proc.poll() is not None:
        print(f"supervisor exited immediately (rc={proc.returncode}); see {run_dir}/supervisor.log")
        return 1
    print(f"started supervisor pid {proc.pid}; log {run_dir}/supervisor.log; events {run_dir}/events.jsonl")
    return 0


def cmd_status(args) -> int:
    run_dir = Path(args.run_dir)
    events = read_jsonl(run_dir / "events.jsonl")
    pid = supervisor_pid(run_dir)
    print(f"supervisor: {'running pid ' + str(pid) if pid else 'not running'}")
    starts = [i for i, e in enumerate(events) if e["kind"] == "run_started"]
    seg = events[starts[-1]:] if starts else events
    if starts:
        e = events[starts[-1]]
        print(f"segment: save={e.get('save')} init={e.get('init')} timesteps={e.get('timesteps')} agent={e.get('agent')}")
    evals = [e for e in seg if e["kind"] == "eval"]
    if evals:
        last = evals[-1]
        print(f"evals: {len(evals)}  best={last.get('best'):.1f}")
        print("last eval: " + fmt_event(last))
    cost = run_dir / "agent_cost.json"
    print(f"agent spend: ${json.loads(cost.read_text())['total_usd']:.3f}" if cost.exists() else "agent spend: $0")
    for d in read_jsonl(run_dir / "decisions.jsonl", 3):
        dec = d.get("decision") if isinstance(d.get("decision"), dict) else {}
        print(f"decision t={d.get('t')} src={d.get('source', 'agent')} action={dec.get('action')} "
              f"param={d.get('param', '')} {d.get('before', '')}->{d.get('after', '')} applied={d.get('applied')} "
              f"reason={str(dec.get('reason', ''))[:120]}")
    notable = [e for e in seg if e["kind"] in NOTIFY]
    for e in notable[-5:]:
        print("event: " + fmt_event(e))
    return 0


def cmd_events(args) -> int:
    for e in read_jsonl(Path(args.run_dir) / "events.jsonl", args.n):
        print(fmt_event(e))
    return 0


def write_control(run_dir: Path, cmd: dict) -> int:
    path = run_dir / "control.json"
    try:
        seq = json.loads(path.read_text()).get("seq", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        seq = 0
    cmd = {"seq": seq + 1, **cmd, "issued_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cmd) + "\n")
    tmp.replace(path)   # atomic: training never reads a half-written file
    print(f"queued seq={cmd['seq']}: {cmd['action']} {cmd.get('param', '')} {cmd.get('value', '')}".rstrip()
          + " (applied at the next rollout boundary; check `status`)")
    return 0


def cmd_adjust(args) -> int:
    if args.param not in HOT_PARAMS:
        print(f"unknown param {args.param!r}; allowed: {', '.join(HOT_PARAMS)}")
        return 2
    lo, hi, integer, _ = HOT_PARAMS[args.param]
    if not lo <= args.value <= hi:
        print(f"{args.param} must be in [{lo}, {hi}]")
        return 2
    value = int(args.value) if integer else args.value
    return write_control(Path(args.run_dir), {"action": "adjust", "param": args.param, "value": value,
                                              "reason": args.reason})


def cmd_stop(args) -> int:
    if not supervisor_pid(Path(args.run_dir)):
        print("not running")
        return 1
    return write_control(Path(args.run_dir), {"action": "stop", "reason": args.reason})


def cmd_kill(args) -> int:
    pid = supervisor_pid(Path(args.run_dir))
    if not pid:
        print("not running")
        return 1
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    print(f"sent SIGTERM to process group of {pid} (no checkpoint is saved)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start", help="launch supervisor.py detached; args after -- go to train_with_jev.py")
    p.add_argument("--sup-arg", dest="sup_args", action="append", default=[],
                   help="extra supervisor.py argument, repeatable (e.g. --sup-arg=--max-restarts=1)")
    p.add_argument("train_args", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    p = sub.add_parser("events")
    p.add_argument("-n", type=int, default=15)
    p = sub.add_parser("adjust")
    p.add_argument("param")
    p.add_argument("value", type=float)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("stop")
    p.add_argument("--reason", required=True)
    sub.add_parser("kill")
    args = ap.parse_args(argv)
    return {"start": cmd_start, "status": cmd_status, "events": cmd_events, "adjust": cmd_adjust,
            "stop": cmd_stop, "kill": cmd_kill}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
