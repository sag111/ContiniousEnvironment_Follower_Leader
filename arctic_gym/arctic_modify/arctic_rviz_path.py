import numpy as np

from src.arctic_gym.arctic_env.arctic_env import ArcticEnv


class ArcticEnvRviz(ArcticEnv):

    def __init__(self,
                 name,
                 discrete_action,
                 time_for_action,
                 trajectory_saving_period,
                 min_distance,
                 max_distance):
        super(ArcticEnvRviz, self).__init__(name,
                                            discrete_action,
                                            time_for_action,
                                            trajectory_saving_period,
                                            min_distance,
                                            max_distance)

        self.leader_position = None
        self.follower_position = None

    def reset(self, move=True) -> np.array:
        self.leader_position, self.follower_position = self._get_pos()
        self.pub.reset_follower_path()
        self.pub.reset_target_path()
        self.pub.reset_target_cam_path()
        obs = super(ArcticEnvRviz, self).reset(move)

        return obs

    def step(self, action: list):
        self.leader_position, self.follower_position = self._get_pos()
        self.pub.update_follower_path(self.follower_position[0], self.follower_position[1])
        self.pub.update_target_path(self.leader_position[0], self.leader_position[1])
        obs, reward, self.done, self.info = super(ArcticEnvRviz, self).step(action)

        return obs, reward, self.done, self.info

    def _get_pos(self):
        """
        Получение информации о позиции, направлении и скорости ведущего и ведомго
        """
        target_odom = self.sub.get_odom_target()
        arctic_odom = self.sub.get_odom()

        leader_position = np.array([np.round(target_odom.pose.pose.position.x, decimals=2),
                                    np.round(target_odom.pose.pose.position.y, decimals=2)])

        follower_position = np.array([np.round(arctic_odom.pose.pose.position.x, decimals=2),
                                      np.round(arctic_odom.pose.pose.position.y, decimals=2)])

        return leader_position, follower_position
