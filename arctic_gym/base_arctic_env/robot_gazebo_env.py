#!/usr/bin/env python
import os
import rospy
import gym

from gym.utils import seeding
from pyhocon import ConfigFactory

from src.instruments.rosmod.subscribers import Subscribers
from src.instruments.rosmod.publishers import Publishers
from src.instruments.rosmod.services import Services
from src.instruments.log.customlogger import logger


PATH_TO_CONFIG = os.path.join(os.getcwd(), 'CONFIG', 'config.conf')
config = ConfigFactory.parse_file(PATH_TO_CONFIG)


log, formatter = logger(name='gazebo_env', level=config.logmode.gazebo_env)


class RobotGazeboEnv(gym.Env):

    def __init__(self):
        self.seed()

        self.srv = Services()
        self.pub = Publishers()
        self.sub = Subscribers()

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
        log.info("Завершение работы Gym окружения")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")
