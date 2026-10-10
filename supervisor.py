"""Run train_with_jev.py, and restart it from a checkpoint when it exits with code 75.

    .venv/bin/python supervisor.py -- --agent claude --save pth/stable_7_jev
    .venv/bin/python supervisor.py --max-restarts 2 -- --agent mock --timesteps 2000000

Everything after `--` goes to train_with_jev.py unchanged, except that on a restart the
supervisor replaces --init (with the saved checkpoint), --timesteps (with what is left) and
--save (adds _r1, _r2, ... so each segment keeps its own _best/_final checkpoints).

Exit codes of the training process:
  0   finished (or stopped by its own degradation rule)  -> supervisor exits 0
  75  restart requested (agent "stop")                    -> run tests, then resume
  *   crash                                               -> record it and exit; no blind restart

The supervisor never touches git or your source files. When the test gate fails it stops and
emits `escalate_to_human`, so a human (or an interactive Claude session) decides what to do.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from agent_bridge import RESTART_EXIT_CODE, RunLog


def set_arg(argv: list[str], flag: str, value) -> list[str]:
    """Return argv with `flag value` replaced (both `--f v` and `--f=v` forms) or appended."""
    out, skip = [], False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok == flag:
            skip = i + 1 < len(argv)
            continue
        if tok.startswith(flag + "="):
            continue
        out.append(tok)
    return out + [flag, str(value)]


def get_arg(argv: list[str], flag: str, default=None):
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default


def default_save(train_args: list[str]) -> str:
    """Mirror train_with_jev.parse_args' default --save."""
    return "pth/stable_7_ctrl" if "--no-jev" in train_args else "pth/stable_7_jev"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-restarts", type=int, default=3)
    # unittest (stdlib) rather than pytest: the project venv does not ship pytest.
    ap.add_argument("--test-cmd", default=f"{shlex.quote(sys.executable)} -m unittest discover -q -s tests",
                    help="must exit 0 before a restart; empty string disables the gate")
    ap.add_argument("--train-script", default="train_with_jev.py")
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="arguments after -- go to the trainer")
    args = ap.parse_args(argv)
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args

    run_dir = Path(get_arg(train_args, "--run-dir", "runs"))
    log = RunLog(run_dir)
    restarts = 0
    pid_file = run_dir / "supervisor.pid"
    pid_file.write_text(f"{os.getpid()}\n")
    log.event("supervisor_started", train_args=train_args, pid=os.getpid())
    try:
        return _loop(args, train_args, run_dir, log, restarts)
    finally:
        pid_file.unlink(missing_ok=True)


def _loop(args, train_args, run_dir, log, restarts) -> int:
    while True:
        cmd = [sys.executable, args.train_script, *train_args]
        rc = subprocess.call(cmd)
        if rc == 0:
            log.event("supervisor_finished", restarts=restarts)
            return 0
        if rc != RESTART_EXIT_CODE:
            log.event("crash", rc=rc, restarts=restarts, cmd=cmd)
            return rc
        restart_file = run_dir / "restart.json"
        info = json.loads(restart_file.read_text())
        restart_file.rename(run_dir / f"restart_{restarts}.json")
        if restarts >= args.max_restarts:
            log.event("escalate_to_human", reason=f"max restarts ({args.max_restarts}) reached", restart=info)
            return RESTART_EXIT_CODE
        if args.test_cmd:
            test = subprocess.run(args.test_cmd, shell=True, capture_output=True, text=True)
            if test.returncode != 0:
                log.event("escalate_to_human", reason="tests failed; not restarting",
                          tail=(test.stdout + test.stderr)[-1500:], restart=info)
                return 1
        requested = int(get_arg(train_args, "--timesteps", info["timesteps_requested"]))
        remaining = requested - int(info["timesteps_done"])
        if remaining <= 0:
            log.event("supervisor_finished", restarts=restarts, note="no timesteps left after stop")
            return 0
        restarts += 1
        # A new --save prefix per segment: otherwise the resumed run's first evaluation
        # (best_score starts at -inf again) would overwrite the previous <save>_best.zip.
        save = get_arg(train_args, "--save") or default_save(train_args)
        save = f"{save.rsplit('_r', 1)[0] if restarts > 1 else save}_r{restarts}"
        train_args = set_arg(set_arg(set_arg(train_args, "--init", info["checkpoint"]),
                                     "--timesteps", remaining), "--save", save)
        log.event("restarting", restart=restarts, init=info["checkpoint"], timesteps=remaining,
                  reason=info.get("reason"))


if __name__ == "__main__":
    sys.exit(main())
