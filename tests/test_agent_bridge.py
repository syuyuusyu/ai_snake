import json
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sb3_contrib import MaskablePPO

import supervisor
import train_ctl
from agent_bridge import (
    RESTART_EXIT_CODE,
    AgentBridgeCallback,
    HumanControlCallback,
    RunLog,
    ScaledSchedule,
    Triggers,
    Tunable,
    build_schema,
    make_claude_advisor,
)


class Box:
    def __init__(self, value):
        self.value = value


def tunables(box):
    return {"ent_coef": Tunable("ent_coef", 0.0, 0.02, lambda: box.value, lambda v: setattr(box, "value", v))}


METRICS = {"score": 100.0, "timeout_rate": 0.0}


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = RunLog(Path(self.tmp.name) / "runs")
        self.eval_cb = SimpleNamespace(n_evals=0, last_metrics=None, best_score=float("-inf"))
        self.box = Box(0.003)
        self.answers = []

    def bridge(self, decision, **kw):
        def advisor(event):
            self.answers.append(event)
            return decision, 0.01
        return AgentBridgeCallback(self.eval_cb, tunables(self.box), advisor, self.log, **kw)

    def new_eval(self, bridge, **metrics):
        self.eval_cb.n_evals += 1
        self.eval_cb.last_metrics = {**METRICS, **metrics}
        self.eval_cb.best_score = max(self.eval_cb.best_score, 100.0)
        bridge._on_rollout_end()
        if bridge.worker is not None:
            bridge.worker.join()

    def events(self):
        return [json.loads(line) for line in (self.log.dir / "events.jsonl").read_text().splitlines()]

    def test_schema_allows_one_param_per_decision(self):
        schema = build_schema(tunables(self.box))
        self.assertEqual(schema["properties"]["param"]["enum"], ["ent_coef"])
        self.assertEqual(schema["required"], ["action", "reason"])
        self.assertFalse(schema["additionalProperties"])

    def test_quiet_evaluation_costs_nothing(self):
        b = self.bridge({"action": "none", "reason": ""})
        self.new_eval(b)
        self.assertEqual(self.answers, [])
        self.assertEqual(self.events()[-1]["kind"], "eval")

    def test_adjust_is_clamped_applied_on_training_thread_and_logged(self):
        b = self.bridge({"action": "adjust", "param": "ent_coef", "value": 9.0, "reason": "x"})
        self.new_eval(b, timeout_rate=0.5)
        self.assertEqual(self.box.value, 0.003)        # not applied from the worker thread
        b._on_rollout_end()
        self.assertEqual(self.box.value, 0.02)         # clamped to the upper bound
        rec = json.loads((self.log.dir / "decisions.jsonl").read_text().splitlines()[-1])
        self.assertEqual((rec["before"], rec["after"], rec["applied"]), (0.003, 0.02, True))
        self.assertAlmostEqual(self.log.total_cost(), 0.01)
        self.assertEqual(self.answers[0]["triggers"], ["timeouts"])
        self.assertIn("ent_coef", self.answers[0]["tunables"])

    def test_cooldown_and_call_limit(self):
        b = self.bridge({"action": "none", "reason": ""}, cooldown_evals=2, max_calls=2)
        for _ in range(6):
            self.new_eval(b, timeout_rate=0.5)
        self.assertEqual(len(self.answers), 2)
        skipped = [e["reason"] for e in self.events() if e["kind"] == "ask_skipped"]
        self.assertIn("cooldown", skipped)
        self.assertIn("max_calls", skipped)

    def test_spend_limit_stops_questions(self):
        self.log.add_cost(5.0)
        b = self.bridge({"action": "none", "reason": ""}, max_usd=2.0)
        self.new_eval(b, timeout_rate=0.5)
        self.assertEqual(self.answers, [])

    def test_degradation_and_plateau_rules(self):
        b = self.bridge({"action": "none", "reason": ""}, triggers=Triggers(plateau_evals=3))
        self.eval_cb.best_score = 100.0
        self.assertIn("degraded", b.triggers_for({"score": 80.0}, 1, 100.0))
        self.assertNotIn("degraded", b.triggers_for({"score": 95.0}, 1, 100.0))
        self.assertIn("plateau", b.triggers_for({"score": 95.0}, 3, 100.0))

    def test_stop_requests_restart_and_halts_steps(self):
        b = self.bridge({"action": "stop", "reason": "needs restart"})
        self.new_eval(b, timeout_rate=0.5)
        self.assertTrue(b._on_step())
        b._on_rollout_end()
        self.assertTrue(b.restart_requested)
        self.assertFalse(b._on_step())

    def test_invalid_decisions_are_recorded_not_applied(self):
        b = self.bridge({"action": "none", "reason": ""})
        for bad in ({"action": "adjust", "param": "gamma", "value": 0.9, "reason": ""},
                    {"action": "adjust", "param": "ent_coef", "value": "big", "reason": ""},
                    {"action": "rm -rf", "reason": ""}, "not a dict"):
            rec = b.apply({"t": 0}, bad)
            self.assertFalse(rec["applied"])
        self.assertEqual(self.box.value, 0.003)

    def test_advisor_exception_becomes_none(self):
        def broken(event):
            raise RuntimeError("boom")
        b = AgentBridgeCallback(self.eval_cb, tunables(self.box), broken, self.log)
        self.new_eval(b, timeout_rate=0.5)
        b._on_rollout_end()
        rec = json.loads((self.log.dir / "decisions.jsonl").read_text().splitlines()[-1])
        self.assertIn("boom", rec["decision"]["reason"])

    def test_scaled_schedule(self):
        s = ScaledSchedule(lambda p: 2.0 * p, 0.5)
        self.assertEqual(s(1.0), 1.0)
        s.scale = 2
        self.assertEqual(s(0.5), 2.0)


class HumanControlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "runs"
        self.log = RunLog(self.run_dir)
        self.box = Box(0.003)

    def ctl(self, *argv):
        return train_ctl.main(["--run-dir", str(self.run_dir), *argv])

    def test_stale_command_is_ignored_and_new_one_applied_once(self):
        self.assertEqual(self.ctl("adjust", "ent_coef", "0.01", "--reason", "old"), 0)
        cb = HumanControlCallback(tunables(self.box), self.log)      # starts after the stale command
        cb._on_rollout_end()
        self.assertEqual(self.box.value, 0.003)
        self.assertEqual(self.ctl("adjust", "ent_coef", "0.005", "--reason", "new"), 0)
        cb._on_rollout_end()
        cb._on_rollout_end()
        self.assertEqual(self.box.value, 0.005)
        recs = [json.loads(line) for line in (self.run_dir / "decisions.jsonl").read_text().splitlines()]
        self.assertEqual(len(recs), 1)
        self.assertEqual((recs[0]["source"], recs[0]["seq"]), ("human", 2))

    def test_ctl_validates_before_writing(self):
        self.assertEqual(self.ctl("adjust", "ent_coef", "0.5", "--reason", "x"), 2)
        self.assertEqual(self.ctl("adjust", "gamma", "0.9", "--reason", "x"), 2)
        self.assertFalse((self.run_dir / "control.json").exists())

    def test_param_missing_from_this_run_is_rejected_by_training(self):
        cb = HumanControlCallback(tunables(self.box), self.log)
        self.ctl("adjust", "jev_budget", "16", "--reason", "x")      # valid name, but not tunable here
        cb._on_rollout_end()
        kinds = [json.loads(line)["kind"] for line in (self.run_dir / "events.jsonl").read_text().splitlines()]
        self.assertIn("control_rejected", kinds)

    def test_stop_command_halts_training(self):
        cb = HumanControlCallback(tunables(self.box), self.log)
        (self.run_dir / "supervisor.pid").write_text(f"{__import__('os').getpid()}\n")   # pretend running
        self.assertEqual(self.ctl("stop", "--reason", "done"), 0)
        cb._on_rollout_end()
        self.assertTrue(cb.stop_requested)
        self.assertFalse(cb._on_step())


class ClaudeAdvisorTest(unittest.TestCase):
    def fake_claude(self, tmp, body):
        path = Path(tmp) / "claude"
        path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return str(path)

    def test_parses_structured_output_and_passes_read_only_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv_file = Path(tmp) / "argv.json"
            exe = self.fake_claude(tmp, f"""
                import json, sys
                json.dump(sys.argv[1:], open({str(argv_file)!r}, "w"))
                print(json.dumps({{"structured_output": {{"action": "none", "reason": "ok"}},
                                   "total_cost_usd": 0.02}}))
            """)
            advisor = make_claude_advisor({"type": "object"}, claude_bin=exe, cwd=tmp)
            decision, cost = advisor({"metrics": {}})
            self.assertEqual((decision["action"], cost), ("none", 0.02))
            argv = json.loads(argv_file.read_text())
            for flag in ("-p", "--json-schema", "--output-format", "--max-budget-usd", "--permission-prompts"):
                self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index("--allowedTools") + 1], "Read")

    def test_failure_or_missing_binary_is_harmless(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = self.fake_claude(tmp, "import sys; print('{\"result\": \"auth error\"}'); sys.exit(1)")
            decision, _ = make_claude_advisor({}, claude_bin=exe, cwd=tmp)({})
            self.assertEqual(decision["action"], "none")
            self.assertIn("auth error", decision["reason"])
            decision, cost = make_claude_advisor({}, claude_bin=str(Path(tmp) / "missing"), cwd=tmp)({})
            self.assertEqual((decision["action"], cost), ("none", 0.0))


class SupervisorTest(unittest.TestCase):
    def test_restart_replaces_init_timesteps_and_save_after_test_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = Path(tmp) / "calls.jsonl"
            run_dir = Path(tmp) / "runs"
            script = Path(tmp) / "fake_train.py"
            script.write_text(textwrap.dedent(f"""
                import json, sys, pathlib
                calls = pathlib.Path({str(calls)!r})
                n = len(calls.read_text().splitlines()) if calls.exists() else 0
                with calls.open("a") as f:
                    f.write(json.dumps(sys.argv[1:]) + "\\n")
                if n == 0:
                    pathlib.Path({str(run_dir)!r}).mkdir(exist_ok=True)
                    (pathlib.Path({str(run_dir)!r}) / "restart.json").write_text(json.dumps(
                        {{"checkpoint": "pth/x_restart.zip", "timesteps_done": 300,
                          "timesteps_requested": 1000, "reason": "t"}}))
                    sys.exit({RESTART_EXIT_CODE})
            """))
            rc = supervisor.main(["--train-script", str(script), "--test-cmd", "true", "--",
                                  "--save", "pth/x", "--timesteps", "1000", "--run-dir", str(run_dir)])
            self.assertEqual(rc, 0)
            second = json.loads(calls.read_text().splitlines()[1])
            self.assertEqual(supervisor.get_arg(second, "--init"), "pth/x_restart.zip")
            self.assertEqual(supervisor.get_arg(second, "--timesteps"), "700")
            self.assertEqual(supervisor.get_arg(second, "--save"), "pth/x_r1")

    def test_failed_tests_block_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs"
            script = Path(tmp) / "fake_train.py"
            script.write_text(textwrap.dedent(f"""
                import json, sys, pathlib
                d = pathlib.Path({str(run_dir)!r}); d.mkdir(exist_ok=True)
                (d / "restart.json").write_text(json.dumps({{"checkpoint": "c.zip", "timesteps_done": 1,
                                                            "timesteps_requested": 10, "reason": "r"}}))
                sys.exit({RESTART_EXIT_CODE})
            """))
            rc = supervisor.main(["--train-script", str(script), "--test-cmd", "false", "--",
                                  "--run-dir", str(run_dir)])
            self.assertEqual(rc, 1)
            kinds = [json.loads(line)["kind"] for line in (run_dir / "events.jsonl").read_text().splitlines()]
            self.assertIn("escalate_to_human", kinds)


class TrainingIntegrationTest(unittest.TestCase):
    """Tiny real run of train_with_jev.main with a scripted advisor."""

    def run_main(self, tmp, decision):
        import train_with_jev
        argv = ["--envs", "1", "--timesteps", "512", "--n-steps", "128", "--batch-size", "64",
                "--eval-freq", "128", "--eval-episodes", "1", "--eval-max-steps", "30", "--no-jev",
                "--horizon", "5", "--device", "cpu", "--agent", "mock", "--agent-trigger", "always",
                "--agent-cooldown", "0", "--save", f"{tmp}/m", "--log-dir", f"{tmp}/logs",
                "--run-dir", f"{tmp}/runs", "--cache", f"{tmp}/cache.jsonl"]
        with patch("train_with_jev.mock_advisor", lambda event: (decision, 0.0)):
            return train_with_jev.main(argv)

    def test_stop_decision_saves_restart_checkpoint_and_exits_75(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = self.run_main(tmp, {"action": "stop", "reason": "test"})
            self.assertEqual(rc, RESTART_EXIT_CODE)
            info = json.loads(Path(f"{tmp}/runs/restart.json").read_text())
            self.assertTrue(Path(info["checkpoint"]).exists())
            self.assertLess(info["timesteps_done"], 512)
            MaskablePPO.load(info["checkpoint"], device="cpu")   # plain load still works

    def test_adjust_decision_changes_learning_rate_multiplier(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = self.run_main(tmp, {"action": "adjust", "param": "lr_scale", "value": 0.5, "reason": "t"})
            self.assertEqual(rc, 0)
            recs = [json.loads(line) for line in Path(f"{tmp}/runs/decisions.jsonl").read_text().splitlines()]
            self.assertEqual((recs[0]["param"], recs[0]["before"], recs[0]["after"]), ("lr_scale", 1.0, 0.5))
            kinds = [json.loads(line)["kind"] for line in Path(f"{tmp}/runs/events.jsonl").read_text().splitlines()]
            self.assertEqual(kinds[0], "run_started")
            self.assertEqual(kinds[-1], "run_finished")


if __name__ == "__main__":
    unittest.main()
