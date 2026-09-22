import unittest

from train_with_ppo import MonitorCallback


class MonitorCallbackTest(unittest.TestCase):
    def test_rollout_metrics_classify_terminal_events(self):
        callback = MonitorCallback()
        infos = [
            {'snake_length': 20, 'step_state': 'hit wall'},
            {'snake_length': 50, 'step_state': 'collied self'},
            {'snake_length': 144, 'step_state': 'Victory'},
            {
                'snake_length': 80,
                'step_state': 'this head approch the food',
                'reward_timeout': -1.0,
            },
            {'snake_length': 75, 'step_state': 'eat food'},
        ]
        dones = [True, True, True, True, False]

        callback._collect_rollout_events(infos, dones)
        metrics = callback._build_rollout_metrics()

        self.assertEqual(metrics['episodes_ended'], 4.0)
        self.assertEqual(metrics['victory_count'], 1.0)
        self.assertEqual(metrics['timeout_count'], 1.0)
        self.assertEqual(metrics['wall_death_count'], 1.0)
        self.assertEqual(metrics['self_death_count'], 1.0)
        self.assertEqual(metrics['death_rate'], 0.5)
        self.assertEqual(metrics['terminal_length_max'], 144.0)
        self.assertEqual(metrics['step_length_max'], 144.0)
        self.assertEqual(metrics['rollout_max_length'], 144.0)

        self.assertEqual(metrics['terminal_count_0_30'], 1.0)
        self.assertEqual(metrics['death_rate_0_30'], 1.0)
        self.assertEqual(metrics['terminal_count_31_72'], 1.0)
        self.assertEqual(metrics['death_rate_31_72'], 1.0)
        self.assertEqual(metrics['terminal_count_73_plus'], 2.0)
        self.assertEqual(metrics['death_rate_73_plus'], 0.0)

    def test_empty_rollout_metrics_are_safe(self):
        metrics = MonitorCallback()._build_rollout_metrics()

        self.assertEqual(metrics['episodes_ended'], 0.0)
        self.assertEqual(metrics['victory_rate'], 0.0)
        self.assertEqual(metrics['death_rate_73_plus'], 0.0)
        self.assertNotIn('terminal_length_mean', metrics)


if __name__ == '__main__':
    unittest.main()
