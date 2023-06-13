#!/usr/bin/env python
import rospy
import gym

from gym.utils import seeding


from src.arctic_gym.topic.subscribers import Subscribers
from src.arctic_gym.topic.publishers import Publishers


from pathlib import Path
from pyhocon import ConfigFactory

project_path = Path(__file__).resolve().parents[3]
config_path = project_path.joinpath('config/config.conf')
config = ConfigFactory.parse_file(config_path)


class RobotGazeboEnv(gym.Env):

    def __init__(self):
        self.seed()

        self.pub = Publishers(config)
        self.sub = Subscribers(config)

        self.episode_num = 0
        self.cumulated_episode_reward = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self, mode="human"):
        raise NotImplementedError()

    def close(self):
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")
