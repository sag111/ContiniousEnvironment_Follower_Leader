import ray
import numpy as np

from pathlib import Path
from gym.spaces import Box
# from pyhocon import ConfigFactory
import json
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.policy_server_input import PolicyServerInput

import src.MyNewModels


SERVER_PATH = Path(__file__).resolve().parent

# CONFIG_PATH = SERVER_PATH.joinpath("config/3c1bc/params.json").__str__()
# CHECKPOINT = SERVER_PATH.joinpath("checkpoints/3c1bc/checkpoint_000040/checkpoint-40").__str__()

CONFIG_PATH = SERVER_PATH.joinpath("config/transformer/params.json").__str__()
CHECKPOINT = SERVER_PATH.joinpath("checkpoints/transformer/checkpoint_000050/checkpoint-50").__str__()

SERVER_ADDRESS = 'localhost'
SERVER_BASE_PORT = 9900


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
        'lasers_sectors_numbers': 48
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

    observation_space = Box(
        np.zeros((10, sensor_config['lasers_sectors_numbers']), dtype=np.float32),
        np.ones((10, sensor_config['lasers_sectors_numbers']), dtype=np.float32)
    )

    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)

    config.update(
        {
            "env": None,
            "env_config": None,
            "observation_space": observation_space,
            "action_space": action_space,
            "input": _input,
            "num_workers": 1,
            "input_evaluation": [],
            "num_gpus": 0,
            "log_level": 'DEBUG',
            "explore": False,
        }
    )

    trainer = PPOTrainer(config=config)
    trainer.restore(CHECKPOINT)

    while True:
        pass
