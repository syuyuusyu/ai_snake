import copy
import io
import json
import random
import tempfile
import threading
import unittest
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure

from game_env import SnakeEnv
from jev_assist import (
    DIRS,
    GateState,
    JevCache,
    LabelBuffer,
    Labeler,
    Snapshot,
    _map_dir,
    _transforms,
    detect_gate,
    make_jev_fn,
    safe_mask_fn,
)
from train_with_jev import (
    JevGateWrapper,
    JevLabelCallback,
    MaskablePPOWithBC,
    make_env,
    parse_args,
)


class JevLabelingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = SnakeEnv(seed=7)
        self.addCleanup(self.env.close)
        self.env.game.snake = deque([(5, 5), (6, 5), (7, 5), (7, 6)])
        self.env.game.food = (2, 3)
        self.env.game.direction = "left"
        self.cache = JevCache(Path(self.tmp.name) / "nested/cache.jsonl", 12, 12)

    def snapshot(self, step_count=0):
        return Snapshot.take(self.env.game, "up", "risk", self.env.game.get_obs(),
                             safe_mask_fn(self.env), step_count=step_count)

    def labeler(self, jev_fn=None, **kwargs):
        labeler = Labeler(lambda: SnakeEnv(seed=1), Mock(), jev_fn, self.cache,
                          horizon=1, **kwargs)
        self.addCleanup(labeler.close)
        return labeler

    def test_wrapper_records_pre_action_state_without_replacing_action(self):
        wrapper = JevGateWrapper(self.env, max_per_rollout=1)
        wrapper.reset()
        self.env.step_count = 17
        snake = list(self.env.game.snake)
        rng_state = self.env.game.rng.getstate()
        with patch("safe_env.detect_gate", return_value="risk"), \
                patch.object(self.env, "step", wraps=self.env.step) as step:
            wrapper.step(0)
            wrapper.step(0)
        self.assertEqual([call.args for call in step.call_args_list], [(0,), (0,)])
        snapshots = wrapper.pop_snapshots()
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].snake, snake)
        self.assertEqual(snapshots[0].step_count, 17)
        self.assertEqual(snapshots[0].rng_state, rng_state)
        self.assertEqual(wrapper.pop_snapshots(), [])

    def test_environment_and_verifier_do_not_modify_global_rng(self):
        before = random.getstate()
        labeler = self.labeler()
        labeler.verify(self.snapshot())
        self.assertEqual(random.getstate(), before)
        self.env.reset(seed=23)
        first = self.env.game.food
        self.env.reset(seed=23)
        self.assertEqual(self.env.game.food, first)
        self.assertEqual(random.getstate(), before)

    def test_rollouts_restore_food_rng_and_timeout_state(self):
        labeler = self.labeler()
        self.env.game.food = (5, 4)
        snap = self.snapshot(step_count=123)
        first, second = labeler._load(snap), labeler._load(snap)
        self.assertIsNot(first.game, second.game)
        self.assertEqual(first.step_count, 123)
        first.step(0)
        second.step(0)
        self.assertEqual(first.game.food, second.game.food)
        self.assertEqual(self.env.game.food, (5, 4))
        self.env.game.food = (2, 3)
        near_timeout = self.snapshot(step_count=12 * 12 * 4 - 1)
        score, failed = labeler._rollout(near_timeout, "up")
        self.assertTrue(failed)
        self.assertLess(score, 0)

    def test_verified_branch_labels_without_jev_and_then_hits_cache(self):
        ask = Mock(return_value="left")
        labeler = self.labeler(ask, audit_rate=0)
        snap = self.snapshot()
        branches = {"up": (-10000, True), "down": (10, False), "left": (9, False)}
        with patch.object(labeler, "_rollout", side_effect=lambda _, move: branches[move]):
            labels = labeler.label([snap])
        self.assertEqual(labels[0][2:], (DIRS.index("down"), 1.0, "rollout"))
        with patch.object(labeler, "verify", side_effect=AssertionError("cache should bypass rollouts")):
            cached = labeler.label([snap])
        self.assertEqual(cached[0][2:], labels[0][2:])
        ask.assert_not_called()

    def test_food_score_margin_can_decide_without_jev(self):
        labeler = self.labeler()
        with patch.object(labeler, "_rollout", side_effect=lambda _, move: (51 if move == "up" else 1, False)):
            move, decisive, _ = labeler.verify(self.snapshot())
        self.assertEqual(move, "up")
        self.assertTrue(decisive)

    def test_risk_gate_detects_entry_into_one_cell_pocket(self):
        game = self.env.game
        game.snake = deque([(7, 6), (6, 6), (6, 7), (6, 8), (7, 8), (8, 8),
                            (8, 7), (9, 7), (9, 6), (9, 5), (9, 4), (9, 3),
                            (9, 2), (9, 1), (9, 0), (8, 0)])
        game.direction, game.food = "right", (11, 11)
        self.assertEqual(detect_gate(game, "down", GateState()), "risk")
        self.assertIsNone(detect_gate(game, "up", GateState()))

    def test_jev_label_keeps_half_weight_after_reload_and_symmetry(self):
        snap = self.snapshot(step_count=31)
        labeler = self.labeler(Mock(return_value="up"), audit_rate=0)
        with patch.object(labeler, "verify", return_value=("left", False, {})):
            labels = labeler.label([snap])
        self.assertEqual(labels[0][2:], (0, 0.5, "jev"))
        reloaded = JevCache(self.cache.path, 12, 12)
        for transform in _transforms(12, 12):
            hit = reloaded.get_label([transform(*p) for p in snap.snake], transform(*snap.food),
                                     _map_dir(transform, snap.direction), snap.step_count)
            self.assertEqual(hit, (_map_dir(transform, "up"), 0.5, "jev"))
        self.assertIsNone(reloaded.get_label(snap.snake, snap.food, snap.direction, 32))

    def test_control_ignores_jev_cache_and_upgrades_with_rollout(self):
        snap = self.snapshot()
        self.cache.put(snap.snake, snap.food, "up", "jev", snap.direction, snap.step_count)
        labeler = self.labeler(None, audit_rate=1, budget_per_round=32)
        with patch.object(labeler, "verify", return_value=("left", True, {})) as verify:
            labels = labeler.label([snap])
        verify.assert_called_once()
        self.assertEqual(labels[0][2:], (2, 1.0, "rollout"))
        self.assertEqual(labeler.stats["jev_calls"], 0)
        self.assertEqual(self.cache.get_label(snap.snake, snap.food, snap.direction, 0),
                         ("left", 1.0, "rollout"))

    def test_control_skips_inconclusive_states(self):
        labeler = self.labeler(None)
        with patch.object(labeler, "verify", return_value=("left", False, {})):
            self.assertEqual(labeler.label([self.snapshot()]), [])
        self.assertEqual(labeler.stats["jev_calls"], 0)

    def test_budget_covers_audits_and_requests(self):
        ask = Mock(return_value="up")
        labeler = self.labeler(ask, audit_rate=1, budget_per_round=2)
        snapshots = [self.snapshot(step_count=i) for i in range(4)]
        with patch.object(labeler, "verify", side_effect=[("up", True, {}), ("left", False, {}),
                                                         ("up", True, {}), ("left", False, {})]):
            labels = labeler.label(snapshots)
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(labeler.stats["jev_calls"], 2)
        self.assertEqual(labeler.stats["audit"], 1)
        self.assertEqual(labeler.stats["audit_agree"], 1)
        self.assertEqual(len(labels), 3)

    def test_parallel_requests_receive_isolated_game_states(self):
        barrier = threading.Barrier(2)
        seen = []

        def ask(game):
            before = (tuple(game.snake), game.food)
            barrier.wait(timeout=5)
            seen.append((before, (tuple(game.snake), game.food)))
            return "up"

        first = self.snapshot()
        self.env.game.food = (1, 2)
        second = self.snapshot()
        labeler = self.labeler(ask, max_workers=2, audit_rate=0, budget_per_round=2)
        with patch.object(labeler, "verify", return_value=("left", False, {})):
            labels = labeler.label([first, second])
        self.assertEqual(len(labels), 2)
        self.assertEqual({before[1] for before, _ in seen}, {(2, 3), (1, 2)})
        self.assertTrue(all(before == after for before, after in seen))

    def test_invalid_answer_or_network_failure_does_not_become_label(self):
        for answer in (Mock(return_value="right"), Mock(side_effect=TimeoutError("offline"))):
            labeler = self.labeler(answer, audit_rate=0)
            with patch.object(labeler, "verify", return_value=("up", False, {})):
                self.assertEqual(labeler.label([self.snapshot()]), [])
            self.assertEqual(labeler.stats["jev_fail"], 1)

    def test_training_http_requests_have_no_hidden_retries(self):
        advisor = make_jev_fn()
        with patch("play_with_jev.urlopen", side_effect=TimeoutError("offline")) as request, \
                patch("play_with_jev.time.sleep") as sleep:
            with self.assertRaises(RuntimeError):
                advisor(self.env.game)
        request.assert_called_once()
        sleep.assert_not_called()

    def test_callback_surfaces_background_failures(self):
        callback = JevLabelCallback(self.labeler(), LabelBuffer())
        self.addCleanup(callback.close)
        callback.future = Future()
        callback.future.set_exception(RuntimeError("label worker failed"))
        with self.assertRaisesRegex(RuntimeError, "label worker failed"):
            callback._harvest()

    def test_callback_deduplicates_snapshots_before_labeling(self):
        labeler = self.labeler()
        callback = JevLabelCallback(labeler, LabelBuffer())
        self.addCleanup(callback.close)
        snapshot = self.snapshot()
        callback.model = Mock()
        callback.model.policy.state_dict.return_value = {}
        callback.model.get_env().env_method.return_value = [[snapshot, copy.deepcopy(snapshot)]]
        with patch.object(labeler, "label", return_value=[]) as label:
            callback._on_rollout_end()
            callback._harvest(wait=True)
        self.assertEqual(len(label.call_args.args[0]), 1)


class JevTrainingTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = make_env(0)()
        self.addCleanup(self.env.close)

    def model(self):
        model = MaskablePPOWithBC("CnnPolicy", self.env, n_steps=4, batch_size=4,
                                  n_epochs=1, learning_rate=1e-3, device="cpu", seed=1)
        model.set_logger(configure(folder=self.tmp.name, format_strings=[]))
        model.bc_min_size = model.bc_steps = 1
        model.bc_batch = 4
        model.bc_buffer = LabelBuffer()
        return model

    def test_jev_weight_halves_bc_loss_even_for_all_jev_batch(self):
        model = self.model()
        obs, _ = self.env.reset()
        mask = safe_mask_fn(self.env)
        action = int(np.flatnonzero(mask)[0])
        original = copy.deepcopy(model.policy.state_dict())
        model.bc_buffer.add([(obs, mask, action, 1.0, "rollout")])
        model.train_bc()
        full_loss = model.logger.name_to_value["bc/loss"]
        self.assertTrue(any(not torch.equal(value, model.policy.state_dict()[key])
                            for key, value in original.items()))
        model.policy.load_state_dict(original)
        model.bc_buffer = LabelBuffer()
        model.bc_buffer.add([(obs, mask, action, 0.5, "jev")])
        model.train_bc()
        self.assertAlmostEqual(model.logger.name_to_value["bc/loss"], full_loss / 2, places=6)
        model._current_progress_remaining = 0.0
        model.train_bc()
        self.assertEqual(model.logger.name_to_value["bc/coef"], 0.05)

    def test_full_background_label_and_bc_training_then_plain_ppo_load(self):
        model = self.model()
        policy = copy.deepcopy(model.policy).to("cpu")
        labeler = Labeler(lambda: SnakeEnv(), policy, Mock(return_value="up"),
                          JevCache(Path(self.tmp.name) / "cache.jsonl", 12, 12),
                          horizon=1, audit_rate=0, budget_per_round=2)
        self.addCleanup(labeler.close)
        callback = JevLabelCallback(labeler, model.bc_buffer, max_per_round=2)
        self.addCleanup(callback.close)
        with patch("safe_env.detect_gate", return_value="loop"), \
                patch.object(labeler, "verify", side_effect=lambda s: (DIRS[int(np.flatnonzero(s.mask)[0])], True, {})):
            model.learn(total_timesteps=16, callback=callback)
        self.assertGreater(len(model.bc_buffer), 0)
        self.assertIn("bc/loss", model.logger.name_to_value)
        self.assertGreater(labeler.stats["decisive"], 0)
        self.assertEqual(labeler.stats["jev_calls"], 0)
        path = Path(self.tmp.name) / "model.zip"
        model.save(path)
        with ZipFile(path) as archive:
            self.assertNotIn("bc_buffer", json.loads(archive.read("data")))
        loaded = MaskablePPO.load(path, env=self.env, device="cpu")
        obs, _ = self.env.reset()
        action, _ = loaded.predict(obs, action_masks=safe_mask_fn(self.env), deterministic=True)
        self.assertTrue(safe_mask_fn(self.env)[int(action)])

    def test_control_defaults_are_separate_and_arguments_validate_early(self):
        control, experiment = parse_args(["--no-jev"]), parse_args([])
        self.assertNotEqual(control.cache, experiment.cache)
        self.assertNotEqual(control.save, experiment.save)
        self.assertEqual((control.board_width, control.board_high), (12, 12))
        for arguments in (["--envs", "0"], ["--audit-rate", "2"], ["--jev-budget", "-1"]):
            with self.subTest(arguments=arguments), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                parse_args(arguments)
