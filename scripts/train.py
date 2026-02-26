import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
# import quad_env
import leg_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
from stable_baselines3.common.utils import set_random_seed
from datetime import datetime
import sys, os
import numpy as np
from typing import Callable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# force CPU usage
env_id = "SingleLeg-v0"
num_cpus = 7

link_length_sum = 0.6

device = "cpu"
torch.set_default_device(device)

from stable_baselines3.common.callbacks import BaseCallback
class MeanEpisodeInfoCallback(BaseCallback):
    """
    Aggregates custom episode info fields across all environments
    and logs their mean to TensorBoard.
    Works with VecMonitor (which provides info['episode']).
    """
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        episodes = [info["episode"] for info in infos if "episode" in info]
        if not episodes:
            return True

        # Collect all keys that appear in episode infos
        keys = set().union(*[ep.keys() for ep in episodes])
        for key in keys:
            vals = [ep[key] for ep in episodes if key in ep]
            if vals:
                mean_val = np.mean(vals)
                self.logger.record(f"custom_mean/{key}", mean_val)

        # flush less often to avoid overlogging
        if self.n_calls % 1000 == 0:
            self.logger.dump(step=self.num_timesteps)
        return True
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function to create a subprocessed env with unique seed.
    """
    def _init():
        # render_mode must be None (or non-human) for SubprocVecEnv
        baseline_xml_path = os.path.join(os.path.dirname(__file__), "single_leg_robstride02.xml")
        env = gym.make(env_id, render_mode=None, xml_path=baseline_xml_path, link_length_sum=link_length_sum)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpus)])
    vec_env = VecMonitor(vec_env, info_keywords=("vel_reward",))

    # set up PPO
    log_dir = f"./logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir, device=device, learning_rate=1e-3)

    extra_log_callback = MeanEpisodeInfoCallback()

    # checkpoint callback: save model every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,               # save every 50k env steps
        save_path=log_dir,
        name_prefix="ppo_single_leg"          # file prefix
    )

    # train with checkpointing and time the training
    start_time = datetime.now()
    model.learn(total_timesteps=5_000_000, callback=[checkpoint_callback, extra_log_callback], progress_bar=True)
    total_time = datetime.now() - start_time

    print(f"Total training time: {total_time}")  # e.g. 1:23:45.678901
    print(f"Total training time (seconds): {total_time.total_seconds():.2f}s")
