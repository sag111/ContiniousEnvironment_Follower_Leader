import gym
import numpy as np

from gym.spaces import Discrete, Box
from scipy.spatial import distance
from src.arctic_gym.utils.reward_constructor import Reward
from src.arctic_gym.arctic_env.arctic_env import ArcticEnv
from src.arctic_gym.gazebo_utils.gazebo_tracker import GazeboLeaderPositionsTracker, GazeboLeaderPositionsTrackerRadar
from ray.tune import register_env


def continuous_radar_env_maker(config):
    name = config["name"]
    env = gym.make(name, **config)

    return env


register_env('ArcticRobot-v2', continuous_radar_env_maker)


class ContinuousRadarEnv(ArcticEnv):

    def __init__(self, name):
        super(ContinuousRadarEnv, self).__init__()

        self.trajectory_saving_period = 8
        self.max_distance = 15.0
        self.min_distance = 6.0
        self.radar_sectors_number = 7
        self.leader_max_speed = 1.0
        self.leader_pos_epsilon = 1.25
        self.max_dev = 2.0
        self.warm_start = 5 / self.config.time_for_action
        self.max_steps = 5000
        self.low_reward = -200
        self.close_coeff = 0.6

        # Информация о ведущем и ведомом
        self.leader_position = None
        self.follower_position = None
        self.follower_orientation = None

        # dataclass наград
        self.reward = Reward()

        if self.config.discrete_action:
            self.action_space = Discrete(3)
        else:
            self.action_space = Box(
                np.array((0, 0.57), dtype=np.float32),
                np.array((1.0, -0.57), dtype=np.float32)
            )

        self.observation_space = Box(
            np.zeros(7, dtype=np.float32),
            np.ones(7, dtype=np.float32)
        )

        self.tracker = GazeboLeaderPositionsTracker(host_object="arctic_robot",
                                                    sensor_name='LeaderTrackDetector',
                                                    saving_period=self.trajectory_saving_period)
        self.leader_history = self.tracker.leader_positions_hist

        self.radar = GazeboLeaderPositionsTrackerRadar(max_distance=self.max_distance,
                                                       host_object="arctic_robot",
                                                       sensor_name='LeaderTrackDetectorRadar',
                                                       position_sequence_length=100,
                                                       detectable_positions='near',
                                                       radar_sectors_number=self.radar_sectors_number)
        self.radar_values = self.radar.radar_values

    def reset(self):
        arctic_obs = super(ContinuousRadarEnv, self).reset()
        self._init_continuous_env_variables()
        self._init_reward_flags()
        obs = self._get_radar_obs(arctic_obs)

        return obs

    def step(self, action):
        arctic_obs, _, _, _ = super(ContinuousRadarEnv, self).step(action)
        obs = self._get_radar_obs(arctic_obs)
        self._is_radar_done(arctic_obs)
        reward = self._compute_radar_reward()
        self.cumulated_radar_episode_reward += reward

        return obs, reward, self.done, self.info

    def _init_continuous_env_variables(self):
        self.green_zone_trajectory_points = list()
        self.left_border_points_list = list()
        self.right_border_points_list = list()
        self.leader_factual_trajectory = list()

        self.tracker.reset()
        self.radar.reset()

        self.step_count = 0
        self.done = False
        self.info = {}

        self.leader_finished = False
        self.saving_counter = 0

        self.cumulated_radar_episode_reward = 0.0

    def _init_reward_flags(self):
        self.is_in_box = False
        self.is_on_trace = False
        self.follower_too_close = False
        self.crash = False

    def _get_radar_obs(self, arctic_obs):
        self.leader_history = self.tracker.scan(arctic_obs[0], arctic_obs[1])
        self.radar_values = self.radar.scan(arctic_obs[1], arctic_obs[2], self.leader_history)

        return self.radar_values.copy()

    def _is_radar_done(self, arctic_obs):
        self.done = False
        self.is_in_box = False
        self.is_on_trace = False

        self.info = {
            "mission_status": "in_progress",
            "agent_status": "moving",
            "leader_status": "moving"
        }

        self._trajectory_in_box()
        self._check_agent_position(arctic_obs[1], arctic_obs[0])

        if self.saving_counter % self.trajectory_saving_period == 0:
            self.leader_factual_trajectory.append(arctic_obs[0])
        self.saving_counter += 1

        if self.step_count > self.warm_start:
            if self.cumulated_episode_reward < self.low_reward:
                self.info["mission_status"] = "fail"
                self.info["leader_status"] = "moving"
                self.info["agent_status"] = "low_reward"
                self.crash = True
                self.done = True

            if np.linalg.norm(arctic_obs[1] - arctic_obs[0]) > self.max_distance:
                self.info["mission_status"] = "fail"
                self.info["leader_status"] = "moving"
                self.info["agent_status"] = "too_far_from_leader"
                self.crash = True
                self.done = True

            if np.linalg.norm(arctic_obs[1] - arctic_obs[0]) < self.min_distance * self.close_coeff:
                self.info["mission_status"] = "fail"
                self.info["leader_status"] = "moving"
                self.info["agent_status"] = "too_close_to_leader"
                self.crash = True
                self.done = True

        self.step_count += 1

        if self.step_count > self.max_steps:
            self.info["mission_status"] = "finished_by_time"
            self.info["leader_status"] = "moving"
            self.info["agent_status"] = "moving"
            self.done = True

    def _compute_radar_reward(self):
        """
        Расчет награды
        """
        reward = 0

        # Ведущий слишком близко
        if self.follower_too_close:
            reward += self.reward.too_close_penalty
        else:
            # В коробке, на маршруте
            if self.is_in_box and self.is_on_trace:
                reward += self.reward.reward_in_box
            # В коробке, не на маршруте
            elif self.is_in_box:
                reward += self.reward.reward_in_dev
            # На маршруте, не в коробке
            elif self.is_on_trace:
                reward += self.reward.reward_on_track
            else:
                # Не на маршруте, не в коробке
                if self.step_count > self.warm_start:
                    reward += self.reward.not_on_track_penalty

        # Авария
        if self.crash:
            reward += self.reward.crash_penalty

        return reward

    def _trajectory_in_box(self):
        self.green_zone_trajectory_points = list()
        accumulated_distance = 0

        for cur_point, prev_point in zip(reversed(self.leader_factual_trajectory[:-1]),
                                         reversed(self.leader_factual_trajectory[1:])):

            accumulated_distance += distance.euclidean(prev_point, cur_point)

            if accumulated_distance <= self.max_distance:
                self.green_zone_trajectory_points.append(cur_point)
            else:
                break

    def _check_agent_position(self, follower_position, leader_position):

        if len(self.green_zone_trajectory_points) > 2:
            closet_point_in_box_id = self.closest_point(follower_position, self.green_zone_trajectory_points)
            closet_point_in_box = self.green_zone_trajectory_points[int(closet_point_in_box_id)]

            closest_green_distance = distance.euclidean(follower_position, closet_point_in_box)

            if closest_green_distance <= self.leader_pos_epsilon:
                self.is_on_trace = True
                self.is_in_box = True

            elif closest_green_distance <= self.max_dev:
                # Агент в пределах дистанции
                self.is_in_box = True
                self.is_on_trace = False

            else:
                closest_point_on_trajectory_id = self.closest_point(follower_position, self.leader_factual_trajectory)
                closest_point_on_trajectory = self.leader_factual_trajectory[int(closest_point_on_trajectory_id)]

                if distance.euclidean(follower_position, closest_point_on_trajectory) <= self.leader_pos_epsilon:
                    self.is_on_trace = True
                    self.is_in_box = False

        if distance.euclidean(leader_position, follower_position) <= self.min_distance:
            self.follower_too_close = True
        else:
            self.follower_too_close = False

    @staticmethod
    def closest_point(point, points, return_id=True):
        """Метод определяет ближайшую к точке точку из массива точек"""
        points = np.asarray(points)
        dist_2 = np.sum((points - point) ** 2, axis=1)

        if not return_id:
            return np.min(dist_2)
        else:
            return np.argmin(dist_2)
