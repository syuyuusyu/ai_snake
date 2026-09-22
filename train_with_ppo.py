import random
from collections import defaultdict

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from game_env import SnakeEnv

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'




repeat_map = defaultdict(int)
bfs_intensity = 1
discount_gamma = 0.995
gae_lambda = 0.97

def make_env(seed=0,board_size=12):
    def _init():
        env = SnakeEnv(
            seed=seed,
            board_size=board_size,
            silent_mode=True,
            bfs_intensity=bfs_intensity,
            reward_gamma=discount_gamma,
        )
        env = ActionMasker(env, SnakeEnv.mask_fn)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

class MonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._reset_rollout_metrics()

    @staticmethod
    def _length_bucket(snake_length):
        if snake_length <= 30:
            return '0_30'
        if snake_length <= 72:
            return '31_72'
        return '73_plus'

    def _reset_rollout_metrics(self):
        self._step_lengths = []
        self._terminal_lengths = []
        self._wall_death_lengths = []
        self._self_death_lengths = []
        self._episode_counts = {
            'ended': 0,
            'death': 0,
            'wall_death': 0,
            'self_death': 0,
            'victory': 0,
            'timeout': 0,
        }
        self._bucket_terminal_counts = {
            '0_30': 0,
            '31_72': 0,
            '73_plus': 0,
        }
        self._bucket_death_counts = {
            '0_30': 0,
            '31_72': 0,
            '73_plus': 0,
        }

    def _collect_rollout_events(self, infos, dones):
        for info, done in zip(infos, dones):
            if 'snake_length' not in info:
                continue

            snake_length = int(info['snake_length'])
            self._step_lengths.append(snake_length)
            if not done:
                continue

            self._terminal_lengths.append(snake_length)
            self._episode_counts['ended'] += 1
            bucket = self._length_bucket(snake_length)
            self._bucket_terminal_counts[bucket] += 1

            step_state = info.get('step_state')
            is_wall_death = step_state == 'hit wall'
            is_self_death = step_state == 'collied self'
            is_victory = step_state == 'Victory'
            is_timeout = bool(info.get('TimeLimit.truncated', False))
            is_timeout = is_timeout or info.get('reward_timeout', 0.0) < 0
            is_death = is_wall_death or is_self_death

            if is_death:
                self._episode_counts['death'] += 1
                self._bucket_death_counts[bucket] += 1
            if is_wall_death:
                self._episode_counts['wall_death'] += 1
                self._wall_death_lengths.append(snake_length)
            if is_self_death:
                self._episode_counts['self_death'] += 1
                self._self_death_lengths.append(snake_length)
            if is_victory:
                self._episode_counts['victory'] += 1
            if is_timeout:
                self._episode_counts['timeout'] += 1

    @staticmethod
    def _add_distribution(metrics, prefix, values):
        if not values:
            return
        values = np.asarray(values, dtype=np.float32)
        metrics[f'{prefix}_mean'] = float(np.mean(values))
        metrics[f'{prefix}_max'] = float(np.max(values))
        metrics[f'{prefix}_p50'] = float(np.percentile(values, 50))
        metrics[f'{prefix}_p90'] = float(np.percentile(values, 90))
        metrics[f'{prefix}_p99'] = float(np.percentile(values, 99))

    def _build_rollout_metrics(self):
        metrics = {}
        self._add_distribution(metrics, 'step_length', self._step_lengths)
        self._add_distribution(metrics, 'terminal_length', self._terminal_lengths)
        self._add_distribution(metrics, 'wall_death_length', self._wall_death_lengths)
        self._add_distribution(metrics, 'self_death_length', self._self_death_lengths)
        if self._step_lengths:
            metrics['rollout_max_length'] = float(max(self._step_lengths))

        ended = self._episode_counts['ended']
        metrics['episodes_ended'] = float(ended)
        for key in ['death', 'wall_death', 'self_death', 'victory', 'timeout']:
            count = self._episode_counts[key]
            metrics[f'{key}_count'] = float(count)
            metrics[f'{key}_rate'] = count / ended if ended else 0.0

        for bucket in ['0_30', '31_72', '73_plus']:
            terminal_count = self._bucket_terminal_counts[bucket]
            death_count = self._bucket_death_counts[bucket]
            metrics[f'terminal_count_{bucket}'] = float(terminal_count)
            metrics[f'death_count_{bucket}'] = float(death_count)
            metrics[f'death_rate_{bucket}'] = (
                death_count / terminal_count if terminal_count else 0.0
            )
        return metrics

    def _on_rollout_start(self) -> None:
        train_info_list = self.training_env.env_method('get_train_info')

        # 初始化计数器
        average_info = {
            'beast_snake_length': 0.0,
            'back_forward_count': 0.0,
            'hit_wall_count': 0.0,
            'collide_self_count': 0.0,
            'repeat_count': 0.0,
            'victory_count': 0.0,
            'rollout_snake_length': 0.0,
        }

        # 环境实例的数量
        num_envs = len(train_info_list)

        # 累加每个环境实例的值
        for train_info in train_info_list:
            for key in average_info:
                average_info[key] += train_info[key]
            # dic = train_info['repeat_map']
            # for k,v in dic.items():
            #     repeat_map[k] = repeat_map[k]+v
                #print(repeat_map)
                
        # 计算平均值
        for key in average_info:
            average_info[key] /= num_envs

        # 打印结果
        for key, value in average_info.items():
            print(f'Average {key}: {value}')
        self.training_env.env_method('reset_rollout')
        self._reset_rollout_metrics()

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        self._collect_rollout_events(infos, dones)
        metric_keys = [
            'reward_food',
            'reward_death',
            'reward_victory',
            'reward_timeout',
            'reward_step',
            'reward_safety',
            'reward_total',
            'safety_before',
            'safety_after',
            'snake_length',
        ]
        for key in metric_keys:
            values = [info[key] for info in infos if key in info]
            if values:
                self.logger.record_mean(f'env/{key}', float(np.mean(values)))
        return True

    def _on_rollout_end(self) -> None:
        for key, value in self._build_rollout_metrics().items():
            self.logger.record(f'snake/{key}', value)


class FixedSeedEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_freq,
        seeds,
        best_model_path,
        max_episode_steps=20_000,
        stop_score_ratio=0.8,
        degradation_patience=2,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.seeds = list(seeds)
        self.best_model_path = best_model_path
        self.max_episode_steps = max_episode_steps
        self.stop_score_ratio = stop_score_ratio
        self.degradation_patience = degradation_patience
        self.best_score = -np.inf
        self.degraded_evaluations = 0
        self.stopped_for_degradation = False
        self.eval_env = None
        self.eval_policy = None

    def _on_training_start(self) -> None:
        self.eval_env = SnakeEnv(
            seed=self.seeds[0],
            board_size=board_size,
            silent_mode=True,
            bfs_intensity=bfs_intensity,
            reward_gamma=discount_gamma,
        )
        self.eval_policy = self.model.policy_class(
            self.model.observation_space,
            self.model.action_space,
            lambda _: 0.0,
            **self.model.policy_kwargs,
        ).to('cpu')
        self._evaluate_and_save()

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            return self._evaluate_and_save()
        return True

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
        self.eval_policy = None

    def _evaluate_and_save(self):
        policy_state = {
            key: value.detach().cpu()
            for key, value in self.model.policy.state_dict().items()
        }
        self.eval_policy.load_state_dict(policy_state)
        self.eval_policy.set_training_mode(False)

        max_lengths = []
        episode_steps = []
        episode_rewards = []
        cause_counts = defaultdict(int)

        for seed in self.seeds:
            random.seed(seed)
            observation, _ = self.eval_env.reset(seed=seed)
            max_length = len(self.eval_env.game.snake)
            total_reward = 0.0

            for step_count in range(1, self.max_episode_steps + 1):
                action, _ = self.eval_policy.predict(
                    observation,
                    action_masks=self.eval_env.mask_fn(),
                    deterministic=True,
                )
                observation, reward, terminated, truncated, info = (
                    self.eval_env.step(int(action))
                )
                total_reward += float(reward)
                max_length = max(max_length, int(info['snake_length']))
                if terminated or truncated:
                    cause = info['step_state'] if terminated else 'timeout'
                    cause_counts[cause] += 1
                    break
            else:
                cause_counts['step_cap'] += 1

            max_lengths.append(max_length)
            episode_steps.append(step_count)
            episode_rewards.append(total_reward)

        max_lengths = np.asarray(max_lengths, dtype=np.float32)
        episode_count = len(self.seeds)
        victory_rate = cause_counts['Victory'] / episode_count
        score = float(np.mean(max_lengths) + board_size ** 2 * victory_rate)
        metrics = {
            'score': score,
            'mean_max_length': float(np.mean(max_lengths)),
            'median_max_length': float(np.median(max_lengths)),
            'p90_max_length': float(np.percentile(max_lengths, 90)),
            'max_length': float(np.max(max_lengths)),
            'mean_episode_steps': float(np.mean(episode_steps)),
            'mean_episode_reward': float(np.mean(episode_rewards)),
            'victory_rate': victory_rate,
            'wall_death_rate': cause_counts['hit wall'] / episode_count,
            'self_death_rate': cause_counts['collied self'] / episode_count,
            'timeout_rate': cause_counts['timeout'] / episode_count,
        }
        for key, value in metrics.items():
            self.logger.record(f'eval/{key}', value)
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            print(
                f"Fixed-seed eval at {self.num_timesteps}: "
                f"score={score:.2f}, mean_max_length={metrics['mean_max_length']:.2f}, "
                f"victory_rate={victory_rate:.2%}"
            )
        if score > self.best_score:
            self.best_score = score
            self.degraded_evaluations = 0
            self.model.save(self.best_model_path)
            if self.verbose:
                print(f'New best model saved to {self.best_model_path}.zip')
        elif score < self.best_score * self.stop_score_ratio:
            self.degraded_evaluations += 1
        else:
            self.degraded_evaluations = 0

        should_continue = self.degraded_evaluations < self.degradation_patience
        if not should_continue:
            self.stopped_for_degradation = True
            if self.verbose:
                print(
                    'Stopping training after repeated fixed-seed evaluation '
                    f'degradation: score={score:.2f}, best={self.best_score:.2f}'
                )
        return should_continue

def schedule_fn(initial_value, final_value=0.0, schedule_type='linear'):

    def scheduler(progress):
        progress = min(max(progress, 0.0), 1.0)
        if schedule_type == 'linear':
            return final_value + progress * (initial_value - final_value)
        elif schedule_type == 'exponential':
            return initial_value * (final_value / initial_value) ** progress
        else:
            raise ValueError("Unsupported schedule type")

    return scheduler


board_size = 12
def main():
    seed_set = set()
    while len(seed_set) < 32:
        seed_set.add(random.randint(0,1e5))
    env = DummyVecEnv([make_env(seed,board_size) for seed in seed_set])
    lr_schedule = schedule_fn(5e-4, 2.5e-6)
    clip_range_schedule = schedule_fn(0.150, 0.025) 
    model = MaskablePPO(
        "CnnPolicy",
        env,
        device=device,
        verbose=1,
        n_steps=512,
        batch_size=512*8,
        n_epochs=4,
        gamma=discount_gamma,
        gae_lambda=gae_lambda,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        ent_coef=0.001,
        tensorboard_log="logs/"
    )
    #checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_snake')
    monitor_callback = MonitorCallback() 
    model.learn(total_timesteps=5e7,callback=[monitor_callback])
    model.save('pth/ppo_snake_early')
    env.close()

def load():
    seed_set = set()
    while len(seed_set) < 128:
        seed_set.add(random.randint(0,1e7))
    env = SubprocVecEnv([make_env(seed,board_size) for seed in seed_set])
    #lr_schedule = schedule_fn(5e-4, 2.5e-6)
    lr_schedule = schedule_fn(3e-6, 1e-6)
    clip_range_schedule = schedule_fn(0.080, 0.030)
    resume_n_steps = 512
    resume_batch_size = resume_n_steps * 8
    model = MaskablePPO.load(
        "pth/stable_4.zip",
        env=env,
        device=device,
        custom_objects={
            'observation_space': env.observation_space,
            'action_space': env.action_space,
            'learning_rate': lr_schedule,
            'lr_schedule': lr_schedule,
            'clip_range': clip_range_schedule,
            # These values must be replaced before _setup_model() creates
            # the rollout buffer. Updating model.n_steps afterwards leaves
            # the loaded buffer at its old size and makes buffer.get() fail.
            'gamma': discount_gamma,
            'gae_lambda': gae_lambda,
            'ent_coef': 0.003,
            'n_epochs': 2,
            'target_kl': 0.01,
            'n_steps': resume_n_steps,
            'batch_size': resume_batch_size,
        },
    )
    model.learning_rate = lr_schedule
    model.lr_schedule = lr_schedule
    model.clip_range = clip_range_schedule
    # Keep the deserialized optimizer and update its parameter groups in place.
    # This avoids passing the Adam-specific ``lr`` keyword to a generic
    # torch.optim.Optimizer constructor.
    for param_group in model.policy.optimizer.param_groups:
        param_group['lr'] = lr_schedule(1.0)
    info_callback = MonitorCallback()
    eval_callback = FixedSeedEvalCallback(
        eval_freq=max(500_000 // env.num_envs, 1),
        seeds=range(32),
        best_model_path='pth/stable_6_best',
    )
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=[info_callback, eval_callback],
        )
        if not eval_callback.stopped_for_degradation:
            model.save('pth/stable_6_final')
    finally:
        env.close()

if __name__ == '__main__':
    load()

    for k,v in repeat_map.items():
        print(f'{k}:{v},')
