import random
import unittest

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.vec_env import DummyVecEnv

from train_with_jev import make_env
from worker_vec_env import WorkerVecEnv


class WorkerVecEnvTest(unittest.TestCase):
    def test_matches_dummy_vec_env_step_for_step(self):
        # A 4x4 board so episodes (wins and timeouts) end within the run.
        seeds = [3, 11, 29, 47, 58]
        dummy = DummyVecEnv([make_env(s, 4, 4, labels=False) for s in seeds])
        worker = WorkerVecEnv([{"seed": s, "board_width": 4, "board_high": 4, "labels": False} for s in seeds],
                              n_workers=2)
        try:
            self.assertTrue(is_masking_supported(worker))
            for vec in (dummy, worker):
                vec.seed(7)
            np.testing.assert_array_equal(dummy.reset(), worker.reset())
            rng, episodes = random.Random(0), 0
            for _ in range(1500):
                masks = get_action_masks(dummy)
                np.testing.assert_array_equal(masks, get_action_masks(worker))
                actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
                d_obs, d_rew, d_done, d_info = dummy.step(actions)
                w_obs, w_rew, w_done, w_info = worker.step(actions)
                np.testing.assert_array_equal(d_obs, w_obs)
                np.testing.assert_allclose(d_rew, w_rew, rtol=1e-6)
                np.testing.assert_array_equal(d_done, w_done)
                for di, wi in zip(d_info, w_info):
                    self.assertEqual(di["TimeLimit.truncated"], wi["TimeLimit.truncated"])
                    self.assertEqual("episode" in di, "episode" in wi)
                    if "episode" in di:
                        episodes += 1
                        self.assertAlmostEqual(di["episode"]["r"], wi["episode"]["r"], places=4)
                        self.assertEqual(di["episode"]["l"], wi["episode"]["l"])
                        np.testing.assert_array_equal(di["terminal_observation"], wi["terminal_observation"])
            self.assertGreater(episodes, 0)
            self.assertEqual(dummy.env_method("get_train_info", indices=[1, 4])[1]["victory_count"],
                             worker.env_method("get_train_info", indices=[1, 4])[1]["victory_count"])
            np.testing.assert_array_equal(get_action_masks(dummy), get_action_masks(worker))
        finally:
            dummy.close()
            worker.close()


if __name__ == "__main__":
    unittest.main()
