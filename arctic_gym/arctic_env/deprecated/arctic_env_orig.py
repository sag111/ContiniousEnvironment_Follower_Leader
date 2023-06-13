import os
import rospy
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from random import choice
from scipy.spatial import distance
from arctic_gym.utils.reward_constructor import Reward
from arctic_gym.base_arctic_env.robot_gazebo_env import RobotGazeboEnv
from arctic_gym.gazebo_utils.gazebo_tracker import GazeboLeaderPositionsTracker, GazeboLeaderPositionsTrackerRadar
from arctic_gym.utils.CustomFormatter import logger
from gym.spaces import Discrete, Box


log, formatter = logger(name='arctic_env', level='INFO')


class ArcticEnv(RobotGazeboEnv):

    def __init__(self, name,
                 discrete_action=False,
                 time_for_action=0.5,
                 trajectory_saving_period=5,
                 leader_max_speed=1.0,
                 min_distance=6.0,
                 max_distance=15.0,
                 leader_pos_epsilon=1.25,
                 max_dev=1.5,
                 max_steps=5000,
                 low_reward=-200,
                 close_coeff=0.6):
        log.debug('init ArcticEnv')
        super(ArcticEnv, self).__init__()

        self.discrete_action = discrete_action
        self.time_for_action = time_for_action
        self.trajectory_saving_period = trajectory_saving_period
        self.leader_max_speed = leader_max_speed
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.leader_pos_epsilon = leader_pos_epsilon
        self.max_dev = max_dev
        self.warm_start = 5 / self.time_for_action
        self.max_steps = max_steps
        self.low_reward = low_reward
        self.close_coeff = close_coeff

        # Периодическое сохранение позиций ведущего в Gazebo
        self.tracker = GazeboLeaderPositionsTracker(host_object="arctic_robot",
                                                    sensor_name='LeaderTrackDetector',
                                                    saving_period=self.trajectory_saving_period)
        self.leader_history = self.tracker.leader_positions_hist

        # Построение радара для observations
        self.radar = GazeboLeaderPositionsTrackerRadar(max_distance=self.max_distance,
                                                       host_object="arctic_robot",
                                                       sensor_name='LeaderTrackDetectorRadar',
                                                       position_sequence_length=100,
                                                       detectable_positions='near',
                                                       radar_sectors_number=7)
        self.radar_values = self.radar.radar_values

        # Информация о ведущем и ведомом
        self.leader_position = None
        self.follower_position = None
        self.follower_orientation = None

        # dataclass наград
        self.reward = Reward()

        if self.discrete_action:
            self.action_space = Discrete(4)
        else:
            self.action_space = Box(
                np.array((0, 0.57), dtype=np.float32),
                np.array((1.0, -0.57), dtype=np.float32)
            )

        self.observation_space = Box(
            np.zeros(7, dtype=np.float32),
            np.ones(7, dtype=np.float32)
        )

        self.targets = [[50.0, -40.0]]

    def reset(self):
        self._reset_sim()
        self._init_env_variables()

        """для reset image data как в step()"""
        self.leader_position, self.follower_position, self.follower_orientation = self._get_positions()
        self.leader_history = self.tracker.scan(self.leader_position, self.follower_position)
        self.radar_values = self.radar.scan(self.follower_position, self.follower_orientation, self.leader_history)

        obs = self._get_obs()
        """"""

        self._safe_zone()
        coord = choice(self.targets)
        self.gz_publishers.move_target(coord[0], coord[1])

        rospy.sleep(0.1)

        self._is_done(self.leader_position, self.follower_position)

        return obs

    def _safe_zone(self):
        first_dots_for_follower_count = int(distance.euclidean(self.follower_position, self.leader_position) * (self.leader_max_speed))

        self.leader_factual_trajectory.extend(zip(np.linspace(self.follower_position[0], self.leader_position[0], first_dots_for_follower_count),
                                                  np.linspace(self.follower_position[1], self.leader_position[1], first_dots_for_follower_count)))

    def _reset_sim(self):
        self._check_all_systems_ready()
        self._set_init_pose()
        self.gazebo.reset_world()
        self._check_all_systems_ready()

    def _check_all_systems_ready(self):
        self.gz_subscribers.check_all_subscribers_ready()
        return True

    def _set_init_pose(self):
        # TODO: телепорты для ведущего и ведомого
        self.gz_publishers.move_base(linear_speed=0.0,
                                     angular_speed=0.0)
        return True

    def _init_env_variables(self):
        """
        Инициализация переменных среды
        """
        # Green Zone
        self.green_zone_trajectory_points = list()
        self.leader_factual_trajectory = list()
        self.follower_factual_trajectory = list()

        # Sensors
        self.tracker.reset()
        self.radar.reset()

        self.cumulated_episode_reward = 0.0
        self._episode_done = False

        self.step_count = 0
        self.done = False
        self.info = {}

        self.leader_finished = False

        self.saving_counter = 0

        self.is_in_box = False
        self.is_on_trace = False
        self.follower_too_close = False
        self.crash = False

        self.code = 0
        self.text = ''

        self.steps_out_box = 0

    def step(self, action):
        log_linear = formatter.colored_logs(round(float(action[0]), 2), 'obs')
        log_angular = formatter.colored_logs(round(float(action[1]), 2), 'obs')
        log.debug(f'Actions: linear - {log_linear}, angular - {log_angular}')
        self._set_action(action)

        """
        Радар по позициям ведущего и ведомого
        """
        self.leader_position, self.follower_position, self.follower_orientation = self._get_positions()
        self.leader_history = self.tracker.scan(self.leader_position, self.follower_position)
        self.radar_values = self.radar.scan(self.follower_position, self.follower_orientation, self.leader_history)

        obs = self._get_obs()
        """"""

        self._is_done(self.leader_position, self.follower_position)

        log_obs = formatter.colored_logs(list(map(lambda x: round(x, 2), obs)), 'obs')
        log.debug(f'Observations: {log_obs}')

        reward = self._compute_reward()
        self.cumulated_episode_reward += reward

        log_reward = formatter.colored_logs(reward, 'yellow')
        if self.is_in_box:
            in_box = formatter.colored_logs(self.is_in_box, 'green')
        else:
            in_box = formatter.colored_logs(self.is_in_box, 'red')
            self.steps_out_box += 1

        if self.is_on_trace:
            on_trace = formatter.colored_logs(self.is_on_trace, 'green')
        else:
            on_trace = formatter.colored_logs(self.is_on_trace, 'red')

        if self.follower_too_close:
            too_close = formatter.colored_logs(self.follower_too_close, 'red')
        else:
            too_close = formatter.colored_logs(self.follower_too_close, 'green')
        log.debug(f'Step reward: {log_reward}, in_box: {in_box}, on_trace: {on_trace}, too_close: {too_close}')

        if self.done:
            log.info(f"Step count: {self.step_count}, steps are out box: {self.steps_out_box}")
            log.info(f"Cumulative_reward: {np.round(self.cumulated_episode_reward, decimals=2)}")

        return obs, reward, self.done, self.info

    # TODO: функция получения и обработки информации с камеры

    def _get_positions(self):
        """
        Получение информации о позиции, направлении и скорости ведущего и ведомго
        """
        target_odom = self.gz_subscribers.get_odom_target()
        arctic_odom = self.gz_subscribers.get_odom()

        leader_position = np.array([np.round(target_odom.pose.pose.position.x, decimals=2),
                                    np.round(target_odom.pose.pose.position.y, decimals=2)])

        follower_position = np.array([np.round(arctic_odom.pose.pose.position.x, decimals=2),
                                      np.round(arctic_odom.pose.pose.position.y, decimals=2)])

        quaternion = arctic_odom.pose.pose.orientation
        follower_orientation_list = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]

        return leader_position, follower_position, follower_orientation_list

    def _get_obs(self):
        """
        Observations среды
        """
        obs_dict = dict()
        obs_dict["LeaderTrackDetector_radar"] = self.radar_values

        return np.array(obs_dict["LeaderTrackDetector_radar"], dtype=np.float32)

    def _set_action(self, action):
        """
        Выбор дискретных или непрерывных действий
        """
        if self.discrete_action:
            if action == 0:
                discrete_action = (0.5, 0.0)
            elif action == 1:
                discrete_action = (0.5, 0.5)
            elif action == 2:
                discrete_action = (0.5, -0.5)
            else:
                discrete_action = (0.0, 0.0)

            self.gz_publishers.move_base(discrete_action[0], discrete_action[1])
            rospy.sleep(self.time_for_action)
        else:
            self.gz_publishers.move_base(action[0], -action[1])
            rospy.sleep(self.time_for_action)

    def _is_done(self, leader_position, follower_position):
        self.done = False
        self.is_in_box = False
        self.is_on_trace = False

        self.info = {
            "mission_status": "in_progress",
            "agent_status": "moving",
            "leader_status": "moving"
        }

        leader_status = self.gz_subscribers.get_target_status()

        try:
            self.code, self.text = leader_status.status_list[-1].status, leader_status.status_list[-1].text
            log.debug(f"Leader status: {self.code}, text: {self.text}")
        except IndexError as e:
            log.debug(f"Leader status trouble: {e}")

        self._trajectory_in_box()
        self._check_agent_position(self.follower_position, self.leader_position)

        if self.saving_counter % self.trajectory_saving_period == 0:
            self.leader_factual_trajectory.append(leader_position)
            self.follower_factual_trajectory.append(follower_position)
        self.saving_counter += 1

        self.step_count += 1

        if self.step_count > self.warm_start:
            # Низкая награда
            if self.cumulated_episode_reward < self.low_reward:
                self.info["mission_status"] = "fail"
                self.info["leader_status"] = "moving"
                self.info["agent_status"] = "low_reward"
                self.crash = True
                self.done = True

                log.error(self.info)

                return 0

            # ведомый далеко от ведущего
            if np.linalg.norm(follower_position - leader_position) > self.max_distance:
                self.info["mission_status"] = "fail"
                self.info["leader_status"] = "moving"
                self.info["agent_status"] = "too_far_from_leader"
                self.crash = True
                self.done = True

                log.error(self.info)

                return 0

            # ведомый слишком близко к ведущему
            if np.linalg.norm(follower_position - leader_position) < self.min_distance * self.close_coeff:
                self.info["mission_status"] = "fail"
                self.info["leader_status"] = "moving"
                self.info["agent_status"] = "too_close_to_leader"
                self.crash = True
                self.done = True

                log.error(self.info)

                return 0

        if self.step_count > self.max_steps:
            self.info["mission_status"] = "finished_by_time"
            self.info["leader_status"] = "moving"
            self.info["agent_status"] = "moving"
            self.done = True

            log.error(self.info)

            return 0

        if self.code == 3 and self.is_in_box:
            self.info["mission_status"] = "success"
            self.info["leader_status"] = "finished"
            self.info["agent_status"] = "finished"
            self.done = True

            log.info(self.info)

            return 0

    def _compute_reward(self):
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
