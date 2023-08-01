import ray
import numpy as np

from pathlib import Path
from gym.spaces import Box
from pyhocon import ConfigFactory
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.policy_server_input import PolicyServerInput

import src.MyNewModels


SERVER_PATH = Path(__file__).resolve().parent
CONFIG_PATH = SERVER_PATH.joinpath("config/FollowerContinuousDyn/params.json").__str__()
CHECKPOINT = SERVER_PATH.joinpath("checkpoints/feats_v14VPC/checkpoint_000300/checkpoint-300").__str__()

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
        'lasers_sectors_numbers': 144
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
        np.zeros((5, sensor_config['lasers_sectors_numbers']), dtype=np.float32),
        np.ones((5, sensor_config['lasers_sectors_numbers']), dtype=np.float32)
    )

    config = ConfigFactory.parse_file(CONFIG_PATH).as_plain_ordered_dict()

    config.update(
        {
            "env": None,
            "env_config": None,
            "observation_space": observation_space,
            "action_space": action_space,
            "input": _input,
            "input_evaluation": [],
            "explore": False,
            "num_gpus": 0,
            "num_workers": 1,
        }
    )

    trainer = PPOTrainer(config=config)
    trainer.restore(CHECKPOINT)

    while True:
        pass
