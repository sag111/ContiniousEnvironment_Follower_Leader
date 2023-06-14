import os
import ray
import numpy as np

from gym.spaces import Box
from pyhocon import ConfigFactory
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.logger import pretty_print
from ray.rllib.examples.custom_metrics_and_callbacks import MyCallbacks


SERVER_ADDRESS = 'localhost'
SERVER_BASE_PORT = 9900
PATH = os.path.join(os.path.dirname(__file__), "config", "FollowerContinuous", "PPO_dyn_obst.conf")
# PATH = os.path.join(os.path.dirname(__file__), "config", "FollowerContinuous", "PPO_obst.conf")
# CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "ppo_featsv2", "checkpoint_000040", "checkpoint-40")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "ppo_featsv2", "checkpoint_000340", "checkpoint-340")
# CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "ppo_featsv2", "checkpoint_000600", "checkpoint-600")
# CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "ppo_featsv2", "checkpoint_000800", "checkpoint-800")


def _input(ioctx):
    if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
        return PolicyServerInput(
            ioctx,
            SERVER_ADDRESS,
            SERVER_BASE_PORT + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
        )
    else:
        return None


if __name__ == '__main__':
    ray.init()

    sensor_config = {
        # 'radar_sectors_number': 7
        # 'radar_sectors_number': 9,
        # 'lidar_sectors_numbers': 21
        'lasers_sectors_numbers': 7
    }

    follower_config = {
        'min_speed': 0,
        'max_speed': 1.0,
        'max_rotation_speed': 0.57
    }

    action_space = Box(
        np.array((follower_config['min_speed'], -follower_config['max_rotation_speed']), dtype=np.float32),
        np.array((follower_config['max_speed'], follower_config['max_rotation_speed']), dtype=np.float32)
    )

    # observation_space = Box(
    #     np.zeros(sensor_config['radar_sectors_number']+sensor_config['lidar_sectors_numbers'], dtype=np.float32),
    #     np.ones(sensor_config['radar_sectors_number']+sensor_config['lidar_sectors_numbers'], dtype=np.float32)
    # )
    # observation_space = Box(
    #     np.zeros(sensor_config['radar_sectors_number'], dtype=np.float32),
    #     np.ones(sensor_config['radar_sectors_number'], dtype=np.float32)
    # )

    observation_space = Box(
        np.zeros(sensor_config['lasers_sectors_numbers'], dtype=np.float32),
        np.ones(sensor_config['lasers_sectors_numbers'], dtype=np.float32)
    )

    configs = ConfigFactory.parse_file(PATH)
    # config = configs["ppo_featsv4"]
    # config["config"]["num_workers"] = 1

    CONFIG = configs["ppo_env4feats12_train5v6"].as_plain_ordered_dict()
    CONFIG["config"]["num_workers"] = 1

    config = {
        "env": None,
        "observation_space": observation_space,
        "action_space": action_space,
        "input": _input,
        "num_workers": 1,
        "input_evaluation": [],
        # "timesteps_per_iteration": 1000,
        # "num_gpus": config['config']['num_gpus'],
        "log_level": 'DEBUG',
        "framework": "torch",
        "normalize_actions": False,
        "explore": False,
    }

    trainer = PPOTrainer(config=config)
    trainer.restore(CHECKPOINT)

    # trainer = ray.rllib.agents.registry.get_trainer_class(config["run"])(config["config"])
    # trainer.restore(CHECKPOINT)

    while True:
        pass
