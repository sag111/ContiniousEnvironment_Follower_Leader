import os
import tf
import json
import rospy
import numpy as np
import requests
import sensor_msgs.point_cloud2 as pc2

from pyhocon import ConfigFactory
from math import atan, tan, sqrt, cos, sin

from src.instruments.log.customlogger import logger
from src.arctic_gym.base_arctic_env.robot_gazebo_env import RobotGazeboEnv
from src.arctic_gym.gazebo_utils.gazebo_tracker import GazeboLeaderPositionsTracker, GazeboLeaderPositionsTrackerRadar


PATH_TO_CONFIG = os.path.join(os.path.expanduser('~'), 'arctic_build', 'robots_HRI', 'config', 'config.conf')
config = ConfigFactory.parse_file(PATH_TO_CONFIG)


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


log, formatter = logger(name='arctic_env', level=config.logmode.arctic_env)


class ArcticEnvPure(RobotGazeboEnv):

    def __init__(self, name,
                 discrete_action=False,
                 time_for_action=0.5,
                 trajectory_saving_period=3,
                 min_distance=6.0,
                 max_distance=25.0):
        log.debug(f'Запуск окружения арктики {name}')
        super(ArcticEnvPure, self).__init__(discrete_action, time_for_action)

        self.min_distance = min_distance
        self.max_distance = max_distance
        self.warm_start = 1

        # Периодическое сохранение позиций ведущего в Gazebo
        self.tracker = GazeboLeaderPositionsTracker(host_object="arctic_robot",
                                                    sensor_name='LeaderTrackDetector',
                                                    saving_period=trajectory_saving_period)

        # Построение радара для observations
        self.radar = GazeboLeaderPositionsTrackerRadar(max_distance=self.max_distance,
                                                       host_object="arctic_robot",
                                                       sensor_name='LeaderTrackDetectorRadar',
                                                       position_sequence_length=100,
                                                       detectable_positions='near',
                                                       radar_sectors_number=7)

        self.leader_history = self.tracker.leader_positions_hist
        self.radar_values = self.radar.radar_values

        # Информация о ведущем и ведомом
        self.follower_orientation = None
        self.leader_position_delta = None

        # Определение конченой точки маршрута
        # self.pts = config.rl_agent.points
        # choice_pt = np.random.randint(len(self.pts))
        # pt = self.pts[choice_pt]
        # self.goal = pt[-2:]
        # self.goal = [68, -32.0]
        self.goal = iter([[68.0, -32.0, -90], [60.0, -73.0, -180], [5.0, -20.0, 90], [12.0, 30.0, 0]])

    def _init_env_variables(self):
        """
        Инициализация переменных среды
        """
        # Sensors
        self.tracker.reset()
        self.radar.reset()

        # is done
        self.step_count = 0
        self.done = False
        self.crash = False
        self.info = {}

        self.history_time = list()
        self.history_twist_x = list()
        self.history_twist_y = list()
        self.delta_time = 0
        self.delta_twist_x = 0
        self.delta_twist_y = 0

        self.theta_camera_yaw = 0

        self.end_stop_count = 0

    def _set_init_pose(self, move):
        """
        Инициализация начального положения
        """
        self.pub.target_cancel_action()
        self.pub.cancel_action()

        # if move:
        #     log.warning('перемещение ведущего и ведомого в начальную позицию')
        #     choice_st = np.random.randint(len(self.pts))
        #     start = self.pts[choice_st]
        #     self.arctic_coords = start[:3]
        #     self.target_coords = start[3:-2]
        #
        #     # Базовые стартовые координаты следования
        #     self.arctic_coords = [5.0, 30.0, 1.0]
        #     self.target_coords = [12.0, 30.0, 1.0]
        #
        #     self.pub.teleport(model="arctic_model", point=self.arctic_coords, quaternion=[0, 0, 0, 1])
        #     self.pub.teleport(model="target_model", point=self.target_coords, quaternion=[0, 0, 0, 1])

        rospy.sleep(0.1)
        self.pub.set_camera_pitch(0)
        self.pub.set_camera_yaw(0)
        rospy.sleep(0.1)

    def reset(self, move=True) -> np.array:
        """
        Инициализирует значения переменных окружения
        Возвращает текущие наблюдения
        """
        self._init_env_variables()
        self._set_init_pose(move)

        self._get_radar()

        self.target = next(self.goal)
        self.pub.move_target(self.target[0], self.target[1], phi=self.target[2])
        log.info(f"Ведущий отправляется в точку {self.target}")

        obs = self._get_obs(self.radar_values)
        self._is_done(self.leader_position_new_phi)

        return obs

    def step(self, action: list) -> (np.array, float, bool, dict):
        """
        Совершает шаг в окружении
        Возвращает наблюдения, награду, флаг завершнения и статус выполнения
        """
        log_linear = formatter.colored_logs(round(float(action[0]), 2), 'obs')
        log_angular = formatter.colored_logs(round(float(action[1]), 2), 'obs')
        log.debug(f'Actions: linear - {log_linear}, angular - {log_angular}')

        self._set_action(action)
        self._get_radar()
        obs = self._get_obs(self.radar_values)
        self._is_done(self.leader_position_new_phi)

        log_obs = formatter.colored_logs(list(map(lambda x: round(x, 2), obs)), 'obs')
        log.debug(f'Наблюдения: {log_obs}')

        return obs, 0.0, self.done, self.info

    def _is_done(self, leader_position):
        """
        Проверка условий выхода из окружения
        """
        follower_position = np.array([0, 0])
        self.done = False

        self.info = {
            "mission_status": "in_progress",
            "agent_status": "moving",
            "leader_status": "moving"
        }

        leader_status = self.sub.get_target_status()

        try:
            self.code, self.text = leader_status.status_list[-1].status, leader_status.status_list[-1].text
            log.debug(f"Статус ведущего: {self.code}, {self.text}")
        except IndexError as e:
            log.debug(f"Проблема получения статуса ведущего: {e}")

        self.step_count += 1

        roll_ang, pitch_ang, _ = tf.transformations.euler_from_quaternion(self.follower_orientation)

        if roll_ang > 1 or roll_ang < -1 or pitch_ang > 1 or pitch_ang < -1:
            self.info["mission_status"] = "fail"
            self.info["agent_status"] = "the_robot_turned_over"
            self.crash = True
            self.done = True
            log.error(self.info)

            return

        if self.step_count > self.warm_start:

            distance = np.linalg.norm(follower_position - leader_position)

            if self.code == 3 and distance < self.min_distance:
                self.info["mission_status"] = "success"
                self.info["leader_status"] = "finished"
                self.info["agent_status"] = "finished"
                self.done = True
                log.info(self.info)

                return

            # ведомый далеко от ведущего
            if distance > self.max_distance:
                self.info["mission_status"] = "fail"
                self.info["leader_status"] = "moving"
                self.info["agent_status"] = "too_far_from_leader"
                self.crash = True
                self.done = True
                log.error(self.info)

                return

            elif self.camera_lead_info['length'] > 0.75 * self.max_distance:
                self.info["mission_status"] = "safety system"
                self.info["leader_status"] = "stop"
                self.info["agent_status"] = "too_far_from_leader_info"
                log.warning(self.info)

                return

            # ведомый слишком близко к ведущему
            if distance < 0.75 * self.min_distance and distance != 0.0:
                self.info["mission_status"] = "warning"
                self.info["leader_status"] = "moving"
                self.info["agent_status"] = "too_close_to_leader"
                log.warning(self.info)

                return

            if not self.camera_leader_information:
                self.info["mission_status"] = "safety system"
                self.info["leader_status"] = "None"
                self.info["agent_status"] = "moving"
                pus_obs = np.zeros(len(self.radar_values), dtype=np.float32)
                if list(self.radar_values[0:len(self.radar_values)]) == list(pus_obs):
                    self.end_stop_count += 1
                    if self.end_stop_count > 25:
                        self.info["mission_status"] = "safety system end"
                        self.done = True
                log.warning(self.info)

                return

        if self.info["leader_status"] == "moving":
            self.end_stop_count = 0

            return

    def _get_radar(self):
        self.follower_orientation, self.follower_delta_position = self._get_positions()
        self.ssd_camera_objects = self._get_ssd_lead_information()
        self.lidar_points = pc2.read_points(self.sub.get_lidar(), skip_nans=False, field_names=("x", "y", "z", "ring"))
        self.length_to_leader = self._calculate_length_to_leader(self.ssd_camera_objects)
        self.camera_lead_info = self._get_camera_lead_info(self.ssd_camera_objects, self.length_to_leader)
        self.leader_position_new_phi = self._get_xy_lead_from_length_phi(self.camera_lead_info)
        self.leader_history = self.tracker.scan(self.leader_position_new_phi, self.follower_delta_position)
        self.radar_values = self.radar.scan([0, 0], self.follower_orientation, self.leader_history)

    def _get_positions(self):
        """
        Получение направления и изменения позиции ведомого
        """
        self.follower_info_odom = self.sub.get_odom()

        quaternion = self.follower_info_odom.pose.pose.orientation
        follower_orientation_list = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]

        follower_time = self.follower_info_odom.header.stamp.to_time()
        follower_linear_x = self.follower_info_odom.twist.twist.linear.x
        follower_linear_y = self.follower_info_odom.twist.twist.linear.y

        # ВРЕМЯ которое используем для поиска дельта С
        self.history_time.append(follower_time)
        if len(self.history_time) > 2:
            self.delta_time = (self.history_time[1]-self.history_time[0])
            log.debug(f"Время шага: {np.round(self.delta_time, decimals=2)}")
            self.history_time.pop(0)

        # Рассчет дельта Х
        self.history_twist_x.append(follower_linear_x)
        if len(self.history_twist_x) > 2:
            self.delta_twist_x = (self.history_twist_x[1] + self.history_twist_x[0])/2
            self.history_twist_x.pop(0)

        # Рассчет дельта Y
        self.history_twist_y.append(follower_linear_y)
        if len(self.history_twist_y) > 2:
            self.delta_twist_y = (self.history_twist_y[1] + self.history_twist_y[0])/2
            self.history_twist_y.pop(0)

        self.delta_cx = self.delta_twist_x * self.delta_time
        self.delta_cx = np.round(self.delta_cx, decimals=2)
        self.delta_cy = self.delta_twist_y * self.delta_time
        self.delta_cy = np.round(self.delta_cy, decimals=2)
        follower_delta_info = {'delta_x': self.delta_cx, 'delta_y': self.delta_cy}

        return follower_orientation_list, follower_delta_info

    def _get_ssd_lead_information(self):
        image = self.sub.get_from_follower_image()
        data = image.data

        results = requests.post(config.object_det.send_data, data=data, timeout=15.0)
        try:
            results = json.loads(results.text)
        except json.decoder.JSONDecodeError:
            log.warning('Пустой JSON от сервиса распознавания объектов')
            results = {}

        return results

    def _calculate_length_to_leader(self, camera_objects):
        max_dist = 25
        length_to_leader = 50
        object_coord = []
        leader_info = next((x for x in camera_objects if x["name"] == "car"), None)

        camera_yaw_state_info = self.sub.get_camera_yaw_state()
        camera_yaw = camera_yaw_state_info.process_value

        if leader_info is not None:
            angles_object = self._calculate_points_angles_objects(leader_info)
            for i in self.lidar_points:
                if i[-1] in [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:

                    if i[0] ** 2 + i[1] ** 2 <= max_dist ** 2 \
                            and (tan(np.deg2rad(-40))+tan(camera_yaw)) * i[0] <= i[1] <= (tan(np.deg2rad(40))+tan(camera_yaw)) * i[0] \
                            and ((tan(angles_object[2])+tan(camera_yaw)) * i[0]) <= i[1] <= ((tan(angles_object[0])+tan(camera_yaw)) * i[0]) \
                            and (tan(angles_object[3]) * i[0]) <= i[2] <= (tan(angles_object[1]) * i[0]):
                        object_coord.append(i)
                        if sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2) <= length_to_leader:
                            length_to_leader = sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2)
        else:
            length_to_leader = None
        log.debug(f'Расстояние до ведущего, определенное с помощью лидара: {length_to_leader}')

        return length_to_leader

    @staticmethod
    def _calculate_points_angles_objects(camera_object):
        xmin = camera_object['xmin']
        ymin = camera_object['ymin']
        xmax = camera_object['xmax']
        ymax = camera_object['ymax']

        xmin = xmin - 20
        xmax = xmax + 20

        p_theta1 = atan((2 * xmin - 640) / 640 * tan(80 / 2))
        p_phi1 = atan(-((2 * ymin - 480) / 480) * tan(64 / 2))  # phi
        p_theta2 = atan((2 * xmax - 640) / 640 * tan(80 / 2))  # theta
        p_phi2 = atan(-((2 * ymax - 480) / 480) * tan(64 / 2))  # phi

        angles_object = [p_theta1, p_phi1, p_theta2, p_phi2]

        return angles_object

    def _get_camera_lead_info(self, camera_objects, length_to_leader):
        results = camera_objects
        info_lead = next((x for x in results if x["name"] == "car"), None)
        log.debug(info_lead)
        self.camera_leader_information = info_lead

        camera_yaw_state_info = self.sub.get_camera_yaw_state()

        camera_yaw = camera_yaw_state_info.process_value

        if bool(info_lead):
            y = (info_lead['xmin'] + info_lead['xmax']) / 2
            x = length_to_leader + 2.1

            hfov = 80
            cam_size = (640, 480)
            theta = np.arctan((2 * y - cam_size[0]) / cam_size[0] * np.tan(hfov / 2))
            _, _, yaw = tf.transformations.euler_from_quaternion(self.follower_orientation)
            theta_new = yaw + theta + camera_yaw
            lead_results = {'length': x, 'phi': theta_new}
            self.theta_camera_yaw = camera_yaw
            self.theta_camera_yaw += theta
            self.pub.set_camera_yaw(self.theta_camera_yaw)
        else:
            x = 0
            _, _, yaw = tf.transformations.euler_from_quaternion(self.follower_orientation)
            theta_new = yaw + camera_yaw
            lead_results = {'length': x, 'phi': theta_new}
            self.theta_camera_yaw = camera_yaw

        return lead_results

    @staticmethod
    def _get_xy_lead_from_length_phi(length_phi):
        length = length_phi['length']
        phi = length_phi['phi']
        lead_x = length * cos(phi)
        lead_y = length * sin(phi)
        results = np.array([np.round(lead_x, decimals=2), np.round(lead_y, decimals=2)])
        return results

    def render(self, mode="human"):
        raise NotImplementedError()
