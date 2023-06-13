import os
import rospy
import numpy as np

from pyhocon import ConfigFactory
from src.arctic_gym.base_arctic_env.robot_gazebo_env import RobotGazeboEnv


PATH = os.path.join(os.path.dirname(__file__), '../config', 'arctic_robot.conf')


class ArcticEnv(RobotGazeboEnv):

    def __init__(self):
        super(ArcticEnv, self).__init__()

        self.config = ConfigFactory.parse_file(PATH)

    def reset(self):
        self._reset_sim()
        self._init_env_variables()
        obs = self._get_obs()

        self.publishers.move_target(50.0, -40.0)

        return obs

    def step(self, action):
        self._set_action(action)
        obs = self._get_obs()
        done = self._is_done()
        info = {}
        reward = self._compute_reward()
        self.cumulated_episode_reward += reward

        return obs, reward, done, info

    def _reset_sim(self):
        self._check_all_systems_ready()
        self._set_init_pose()
        self.gazebo.reset_world()
        self._check_all_systems_ready()

    def _check_all_systems_ready(self):
        self.subscribers.check_all_subscribers_ready()
        return True

    def _set_init_pose(self):
        # TODO: добавить телепорты для ведущго и ведомого
        self.publishers.move_base(linear_speed=0.0, angular_speed=0.0)
        return True

    def _init_env_variables(self):
        self.cumulated_episode_reward = 0.0

    def _get_obs(self):
        target_odom = self.subscribers.get_odom_target()
        arctic_odom = self.subscribers.get_odom()

        leader_position = np.array([np.round(target_odom.pose.pose.position.x, decimals=2),
                                   np.round(target_odom.pose.pose.position.y, decimals=2)])

        follower_position = np.array([np.round(arctic_odom.pose.pose.position.x, decimals=2),
                                      np.round(arctic_odom.pose.pose.position.y, decimals=2)])

        quaternion = arctic_odom.pose.pose.orientation
        follower_orientation_list = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])

        return leader_position, follower_position, follower_orientation_list

    def _set_action(self, action):
        if self.config.discrete_action:
            if action == 0:
                discrete_action = (1.0, 0.0)
            elif action == 1:
                discrete_action = (0.5, 0.5)
            elif action == 2:
                discrete_action = (0.5, -0.5)
            else:
                discrete_action = (0.0, 0.0)
            self.publishers.move_base(discrete_action[0], discrete_action[1])
            rospy.sleep(self.config.time_for_action)
        else:
            self.publishers.move_base(action[0], action[1])
            rospy.sleep(self.config.time_for_action)

    def _is_done(self):
        return False

    def _compute_reward(self):
        return 1.0
