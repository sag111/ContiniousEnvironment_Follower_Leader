#!/usr/bin/env python
import os.path

import rospy

from src.arctic_gym.arctic_env.arctic_env import arctic_env_maker
from ray.rllib.agents.ppo import PPOTrainer

PATH = os.path.join(os.path.dirname(__file__), "config", "FollowerContinuous", "PPO_obst.conf")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "ppo_featsv2", "checkpoint_000040", "checkpoint-40")


if __name__ == '__main__':
    rospy.init_node('arctic_gym_rl', anonymous=True, log_level=rospy.INFO)

    env_config = {"name": "ArcticRobot-v1"}
    env = arctic_env_maker(env_config)

    config = {
        "env": "ArcticRobot-v1",
        "timesteps_per_iteration": 1000,
        "num_workers": 0,
        "log_level": 'DEBUG',
        "framework": "torch",
        "normalize_actions": False,
        "env_config": env_config
    }

    trainer = PPOTrainer(config=config)
    trainer.restore(CHECKPOINT)

    while True:
        obs = env.reset()
        done = False
        info = {}
        while not done:
            action = trainer.compute_single_action(obs, explore=False)

            action[0] /= 2

            obs, reward, done, info = env.step(action)

            rospy.logwarn(obs),
            rospy.logerr(action)

        rospy.logerr(info)

    env.reset()
    env.close()
