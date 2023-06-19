import open3d
import pdal

import os
import time

import tf
import threading
import json
import pandas as pd
import rospy
import numpy as np
import requests
import sensor_msgs.point_cloud2 as pc2

import matplotlib.pyplot as plt

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from pyhocon import ConfigFactory
from math import atan, tan, sqrt, cos, sin
from random import choice
from gym.spaces import Discrete, Box
from scipy.spatial import distance

# from src.instruments.log.customlogger import logger

from src.arctic_gym.utils.reward_constructor import Reward
from src.arctic_gym.base_arctic_env.robot_gazebo_env import RobotGazeboEnv
from src.arctic_gym.gazebo_utils.gazebo_tracker import GazeboLeaderPositionsTracker_v2, GazeboLeaderPositionsCorridorLasers



from src.arctic_gym.utils.misc import rotateVector

PATH_TO_CONFIG = os.path.join(os.getcwd(), 'CONFIG', 'config.conf')
config = ConfigFactory.parse_file(PATH_TO_CONFIG)

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# log, formatter = logger(name='arctic_env', level=config.logmode.arctic_env)


class ArcticEnv(RobotGazeboEnv):

    def __init__(self, name,
                 discrete_action=False,
                 time_for_action=0.5,
                 trajectory_saving_period=3,
                 leader_max_speed=1.0,
                 min_distance=6.0,
                 max_distance=25.0,
                 leader_pos_epsilon=1.25,
                 max_dev=1.5,
                 max_steps=5000,
                 low_reward=-200,
                 close_coeff=0.6):
        print('Запуск окружения арктики')
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
        # TODO traker_v2 for history and corridor
        self.tracker_v2 = GazeboLeaderPositionsTracker_v2(host_object="arctic_robot",
                                                          sensor_name='LeaderTrackDetector',
                                                          saving_period=self.trajectory_saving_period)

        self.leader_history_v2 = self.tracker_v2.leader_positions_hist
        self.corridor_v2 = self.tracker_v2.corridor


        self.laser = GazeboLeaderPositionsCorridorLasers(max_distance=self.max_distance,
                                                         host_object="arctic_robot",
                                                         sensor_name='LeaderCorridor_lasers',
                                                         react_to_green_zone=True,
                                                         react_to_safe_corridor=True,
                                                         react_to_obstacles=False,
                                                         front_lasers_count=5,
                                                         back_lasers_count=2,
                                                         position_sequence_length=100)

        self.laser_values = self.laser.laser_values_obs
        print(self.laser_values)

        # Информация о ведущем и ведомом
        self.leader_position = None
        self.follower_position = None
        self.follower_orientation = None


        self.leader_position_delta = None
        self.follower_position_delta = [0, 0]

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

        self.pts = config.rl_agent.points

        choice_pt = np.random.randint(len(self.pts))
        # choice_pt = 10

        pt = self.pts[choice_pt]

        self.goal = pt[-2:]

    def reset(self, move=True):

        self.pub.set_camera_pitch(0)
        self.pub.set_camera_yaw(0)
        rospy.sleep(0.1)

        self.pub.reset_follower_path()
        self.pub.reset_target_path()
        self.pub.reset_target_cam_path()

        if move:
            print('перемещение ведущего и ведомого в начальную позицию')
            # self.arctic_coords = [0.0, 30.0, 1.0]
            # self.target_coords = [12.0, 30.0, 1.0]

            self.arctic_coords = [0.0, -10.0, 1.0]
            self.target_coords = [12.0, -10.0, 1.0]

            # self.arctic_coords = [40.0, 0.0, 1.0]
            # self.target_coords = [47.0, 0.0, 1.0]

            self.pub.teleport(model="arctic_model", point=self.arctic_coords, quaternion=[0, 0, 0, 1])
            self.pub.teleport(model="target_model", point=self.target_coords, quaternion=[0, 0, 0, 1])

        # self._reset_sim()
        self._init_env_variables()
        self.follower_delta_position = self._get_delta_position()

        """для reset image data как в step()"""
        self.leader_position, self.follower_position, self.follower_orientation = self._get_positions()
        self.roll_ang, self.pitch_ang, self.yaw_ang = tf.transformations.euler_from_quaternion(
            self.follower_orientation)

        # Вызов основных функций
        # Получение информации с SSD object detection
        self.ssd_camera_objects = self._get_ssd_lead_information(self.sub.get_from_follower_image())
        # Получение точек лидара
        self._get_lidar_points()
        # Получение расстояния до лидера и облако точек без точек лидера
        self.length_to_leader, self.other_points = self._calculate_length_to_leader(self.ssd_camera_objects)
        # Получение информации о лидере с камеры
        self.camera_lead_info = self._get_camera_lead_info(self.ssd_camera_objects, self.length_to_leader)
        # Сопоставление и получение координат ведущего на основе информации о расстояние и угле отклонения
        self.leader_position_new_phi = self._get_xy_lead_from_length_phi(self.camera_lead_info)


        # Получение истории и корридора
        self.leader_history_v2, self.corridor_v2 = self.tracker_v2.scan(self.leader_position_new_phi,
                                                                        self.follower_position,
                                                                        self.follower_delta_position)
        # Получение точек препятствий и формирование obs
        self.cur_object_points_1, self.cur_object_points_2 = self._get_obs_points(self.other_points)
        self.laser_values = self.laser.scan([0.0, 0.0], self.follower_orientation, self.leader_history_v2,
                                            self.corridor_v2, self.cur_object_points_1, self.cur_object_points_2)

        obs = self._get_obs()
        """"""
        self._safe_zone()
        self.pub.move_target(self.goal[0], self.goal[1])
        self._is_done(self.leader_position, self.follower_position)

        return obs

    def _safe_zone(self):
        first_dots_for_follower_count = int(distance.euclidean(self.follower_position, self.leader_position) * (self.leader_max_speed))

        self.leader_factual_trajectory.extend(zip(np.linspace(self.follower_position[0], self.leader_position[0], first_dots_for_follower_count),
                                                  np.linspace(self.follower_position[1], self.leader_position[1], first_dots_for_follower_count)))

    def _reset_sim(self):
        self._check_all_systems_ready()
        self._set_init_pose()
        self.srv.reset_world()
        self._check_all_systems_ready()

    def _check_all_systems_ready(self):
        self.sub.check_all_subscribers_ready()
        return True

    def _set_init_pose(self):
        # TODO: телепорты для ведущего и ведомого
        self.pub.move_base(linear_speed=0.0,
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

        self.tracker_v2.reset()

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

        self.history_time = list()
        self.delta_time = 0

        self.history_twist_x = list()
        self.delta_twist_x = 0

        self.history_twist_y = list()
        self.delta_twist_y = 0

        self.theta_camera_yaw = 0


        self.leader_pose_from_cam_glob = list()
        self.all_delta_pose = [0, 0]

        self.end_stop_count = 0

        # для тестов
        self.count_leader_steps_reward = 1
        self.count_leader_reward = 0
        self.leader_finish = False
        self.count_stop_leader = 0

    def step(self, action):

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        log_linear = round(float(action[0]), 2)
        log_angular = round(float(action[1]), 2)
        print(f'Actions: linear - {log_linear}, angular - {log_angular}')
        self._set_action(action)
        self.follower_delta_position = self._get_delta_position()
        """
        Радар по позициям ведущего и ведомого
        """
        self.leader_position, self.follower_position, self.follower_orientation = self._get_positions()
        self.roll_ang, self.pitch_ang, self.yaw_ang = tf.transformations.euler_from_quaternion(self.follower_orientation)
        self.pub.update_follower_path(self.follower_position[0], self.follower_position[1])
        self.pub.update_target_path(self.leader_position[0], self.leader_position[1])
        # Вызов основных функций
        # Получение информации с SSD object detection
        self.ssd_camera_objects = self._get_ssd_lead_information()
        # Получение точек лидара
        self._get_lidar_points()
        # Получение расстояния до лидера и облако точек без точек лидера
        self.length_to_leader, self.other_points = self._calculate_length_to_leader(self.ssd_camera_objects)
        # Получение информации о лидере с камеры
        self.camera_lead_info = self._get_camera_lead_info(self.ssd_camera_objects, self.length_to_leader)
        # Сопоставление и получение координат ведущего на основе информации о расстояние и угле отклонения
        self.leader_position_new_phi = self._get_xy_lead_from_length_phi(self.camera_lead_info)


        # Получение истории и корридора
        self.leader_history_v2, self.corridor_v2 = self.tracker_v2.scan(self.leader_position_new_phi,
                                                                        self.follower_position,
                                                                        self.follower_delta_position)

        # Получение точек препятствий и формирование obs
        self.cur_object_points_1, self.cur_object_points_2 = self._get_obs_points(self.other_points)
        self.laser_values = self.laser.scan([0.0, 0.0], self.follower_orientation, self.leader_history_v2,
                                            self.corridor_v2, self.cur_object_points_1, self.cur_object_points_2)

        obs = self._get_obs()
        """"""
        self._is_done(self.leader_position, self.follower_position)
        log_obs = list(map(lambda x: round(x, 2), obs))
        print(f'Наблюдения: {log_obs}')
        reward = self._compute_reward()
        self.cumulated_episode_reward += reward
        print(self.cumulated_episode_reward)

        log_reward = reward
        if self.is_in_box:
            in_box = self.is_in_box
        else:
            in_box = self.is_in_box
            self.steps_out_box += 1

        if self.is_on_trace:
            on_trace = self.is_on_trace
        else:
            on_trace = self.is_on_trace

        if self.follower_too_close:
            too_close = self.follower_too_close
        else:
            too_close = self.follower_too_close
        print(f'Награда за шаг: {log_reward}, в зоне следования: {in_box}, на пути: {on_trace}, слишком близко: {too_close}')

        if self.done:
            print(f"Количество шагов: {self.step_count}, шагов вне зоны следования: {self.steps_out_box}")
            print(f"Общая награда: {np.round(self.cumulated_episode_reward, decimals=2)}")
            # self._test_sys()

        return obs, reward, self.done, self.info

    # TODO: функция получения и обработки информации с камеры

    def _test_sys(self):
        start_target = self.target_coords
        start_arctic = self.arctic_coords
        goal = self.goal
        step_count = self.step_count
        steps_out = self.steps_out_box
        reward = np.round(self.cumulated_episode_reward, decimals=2)
        info = self.info
        leader_reward = self.count_leader_reward
        leader_steps = self.count_leader_steps_reward

        stop_count = self.count_stop_leader

        d = {'col1': [start_target],
             'col2': [start_arctic],
             'col3': [goal],
             'col4': [step_count],
             'col5': [steps_out],
             'col6': [reward],
             'col7': [info],
             'col8': [leader_reward],
             'col9': [leader_steps],
             'col10': [stop_count]}
        df = pd.DataFrame(data=d)

        df.to_csv("~/arctic_build/robots_HRI/arctic_gym/arctic_env/data/steps_with_all.csv", index=False, mode='a',
                  header=False)
        #
        # df.to_csv("~/arctic_build/robots_HRI/arctic_gym/arctic_env/data/steps_without_9.csv", index=False, mode='a',
        #           header=False)

    def _get_positions(self):
        """
        Получение информации о позиции, направлении и скорости ведущего и ведомго
        """
        target_odom = self.sub.get_odom_target()
        arctic_odom = self.sub.get_odom()

        leader_position = np.array([np.round(target_odom.pose.pose.position.x, decimals=2),
                                    np.round(target_odom.pose.pose.position.y, decimals=2)])

        follower_position = np.array([np.round(arctic_odom.pose.pose.position.x, decimals=2),
                                      np.round(arctic_odom.pose.pose.position.y, decimals=2)])

        quaternion = arctic_odom.pose.pose.orientation
        follower_orientation_list = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]

        return leader_position, follower_position, follower_orientation_list

    @staticmethod
    def _get_leader_local_pose_info(leader_position, follower_position):
        local_lead_pose = leader_position - follower_position
        return local_lead_pose

    @staticmethod
    def _get_length_phi_lead_from_xy(leader_pose_local):
        lead_x = leader_pose_local[0]
        lead_y = leader_pose_local[1]
        length = sqrt(lead_x**2 + lead_y**2)
        phi = atan(lead_y/lead_x)
        results = {'length': length, 'phi': phi}
        return results

    @staticmethod
    def _get_xy_lead_from_length_phi(length_phi):
        """
        функция вычисления координат ведущего относительно ведомого на основе информации о расстоянии и угле отклонения
        Args:
            length_phi = ['length': 12, 'phi' : 0.1]
        Returns:
            Локальные координаты ведущего
            [x, y]
        """
        length = length_phi['length']
        phi = length_phi['phi']
        lead_x = length * cos(phi)
        lead_y = length * sin(phi)
        results = np.array([np.round(lead_x, decimals=5), np.round(lead_y, decimals=5)])
        return results

    @staticmethod
    def _get_four_points(x):
        """
        Функция округления и получения соседней точки в непосредственной близости от полученной
        """
        coeff = 0.1
        a1 = [np.round(x[0], decimals=1), np.round(x[1], decimals=1)]
        a2 = [np.round(x[0] + coeff, decimals=1), np.round(x[1], decimals=1)]
        return a1, a2

    @staticmethod
    def _calculate_points_angles_objects(camera_object):
        """
        функция вычисления углов по значениям bounding box
        Args:
            camera_object = ['xmin': 120, 'ymin' : 100, 'xmax': 300 , 'ymax':400 ]
        Returns:
            Углы для ориентации объекта
            [p_theta1, p_phi1, p_theta2, p_phi2]
        """
        # camera_object = next((x for x in camera_object if x["name"] == "car"), None)
        xmin = camera_object['xmin']
        ymin = camera_object['ymin']
        xmax = camera_object['xmax']
        ymax = camera_object['ymax']

        # xcent = (xmin + xmax)/2
        # ycent = (ymin + ymax)/2

        xmin = xmin - 20
        xmax = xmax + 20
        # ymin = ycent - 20
        # ymax = ycent + 20

        p_theta1 = atan((2 * xmin - 640) / 640 * tan(80 / 2))
        p_phi1 = atan(-((2 * ymin - 480) / 480) * tan(64 / 2))  # phi
        p_theta2 = atan((2 * xmax - 640) / 640 * tan(80 / 2))  # theta
        p_phi2 = atan(-((2 * ymax - 480) / 480) * tan(64 / 2))  # phi

        angles_object = [p_theta1, p_phi1, p_theta2, p_phi2]

        return angles_object


    def _get_ssd_lead_information(self, image):
        """
        функция отправляющая изображение с камеры на Flask сервер с object detection и получающая результаты
        детектирования объектов
        self.sub.get_from_follower_image() - обращается к топику ROS за изображением с камеры

        Пользователю необходимо передавать в данную функцию собственной изображение в собственную модель
        детектирования объектов посредством post запроса.
        Метод используется в reset и step, в тех местах пользователю необходимо изменить подачу изображения или прописать
        собственный топик ROS.

        Args:
            image = self.sub.get_from_follower_image()
        Returns:
            Результат детектирования
            camera_objects = {}
        """
        # image = self.sub.get_from_follower_image()
        data = image.data

        # results = requests.post(config.object_det.send_data, data=data, timeout=15.0)
        results = requests.post('http://192.168.1.35:3333/detection', data=data, timeout=15.0)
        # results = requests.post('http://localhost:6789/detection', data=data, timeout=15.0)

        try:
            results = json.loads(results.text)
        except json.decoder.JSONDecodeError:
            print('Пустой JSON от сервиса распознавания объектов')
            results = {}
        camera_objects = results
        return camera_objects


    def _get_lidar_points(self):
        """
        функция обращается к топику лидара робота и получает облако точек.
        Необходимо прописать топик для подключения к лидару ROS Gazebo.
        """
        lidar = self.sub.get_lidar()
        self.lidar_points = pc2.read_points(lidar, skip_nans=False, field_names=("x", "y", "z", "ring"))

    def _get_obs_points(self, points_list):
        """
        Функция обрабатывающая облако точек лидара для веделения препятствий. Первоначально функция фильтрует точки
        поверхности методом CSF (Cloth Simulation Filter), оставляя только точки препятствий. Далее, нормализует их
        относительно локального положения ведомого. После, полученный список проэцируется на 2D плоскость, уменьшается
        дискретность и округляются значения координат оставшихся точек. Список добавляется вторым соседними мнимыми точками
        в непосредственной близости для формирования отрезков, которые в дальнейшем проверяются лидарными признаками
        нейросетевой моделью.
        Args:
            points_list - облако точек лидара полунные в _get_lidar_points
            points_list : pc2.read_points(lidar, skip_nans=False, field_names=("x", "y", "z", "ring"))

        Returns:
            Списки спроецированных точек препятствий.
            fil_ob_1 = np.array()
            fil_ob_2 = np.array()
        """
        print('OTHER POINTS', len(points_list)) # массив
        t1 = time.time()

        #### Фильтрация, получение точек и передача их в класс PointCloud
        open3d_cloud = open3d.geometry.PointCloud()
        # TODO : Исправить (подумать над альтернативой + оптимизация)
        max_dist = self.laser.laser_length
        # max_dist = 4
        # Отсекание облака точек за пределами удаленности от робота на расстоянии 4 метра
        xyz = [(x, y, z) for x, y, z in points_list if x**2 + y**2 <= max_dist**2]  # get xyz
        print('OTHER POINTS in radius', len(xyz))

        if len(xyz) > 0:
            t2 = time.time()
            # Запись усеченного облака точек в файл формата pcd
            open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
            open3d.io.write_point_cloud("test_pdal_1.pcd", open3d_cloud)
            # Инференс запуска сегментации рельефа поверхности
            pc = (
                    pdal.Reader.pcd("test_pdal_1.pcd")
                    | pdal.Filter.csf(ignore="Classification[7:7]", threshold=0.6)
                    | pdal.Filter.range(limits="Classification[1:1]")  # препятсвтвия
                    # | pdal.Filter.range(limits="Classification[2:2]")  # земля
            )
            pc.execute()
            arr_fil = pc.arrays[0]
            print('after filtering', len(arr_fil))

            # Окруление точек и запись их в список
            list_to_arr = list()
            for i in range(len(arr_fil)):
                list_to_arr.append([np.round(arr_fil[i][0], decimals=1),
                                    np.round(arr_fil[i][1], decimals=1), 0])
            t2_end = time.time() - t2

            # получение списка неповторяющихся точек в проекции на 2D
            list_fil = list()
            list_fil_2 = list()
            yaw = np.degrees(self.yaw_ang)
            for item in list_to_arr:
                if item not in list_fil:
                    list_fil.append(item)
                    ob_point = rotateVector(np.array([item[0], item[1]]), yaw)
                    list_fil_2.append([ob_point[0], ob_point[1]])

            fil_ob_1 = list()
            fil_ob_2 = list()
            for i in list_fil_2:
                # TODO: исправить соединение точек (оптимизировать, оставить один список)
                p1, p2 = self._get_four_points(i)

                fil_ob_1.append(p1)
                fil_ob_2.append(p2)
        else:
            fil_ob_1 = []
            fil_ob_2 = []

        print("TIME FILTERING", time.time() - t1)
        # self._test_lid_filters(len(xyz), len(arr_fil), t2_end, time.time() - t1)

        return fil_ob_1, fil_ob_2

    def _test_lid_filters(self, all_points, after_filter, time_fil, all_time_fil):

        start_target = self.target_coords
        start_arctic = self.arctic_coords
        goal = self.goal
        step_count = self.step_count
        steps_out = self.steps_out_box
        reward = np.round(self.cumulated_episode_reward, decimals=2)
        info = self.info
        leader_reward = self.count_leader_reward
        leader_steps = self.count_leader_steps_reward
        stop_count = self.count_stop_leader





        d = {'col1': [start_target],
             'col2': [start_arctic],
             'col3': [goal],
             'col4': [step_count],
             'col5': [steps_out],
             'col6': [reward],
             'col7': [info],
             'col8': [leader_reward],
             'col9': [leader_steps],
             'col10': [stop_count],
             'col11': [all_points],
             'col12': [after_filter],
             'col13': [time_fil],
             'col14': [all_time_fil],
             }

        df = pd.DataFrame(data=d)
        df.to_csv("~/arctic_build/robots_HRI/arctic_gym/arctic_env/data_lid/csf.csv", index=False, mode='a',
                  header=False)

        return 0

    def _calculate_length_to_leader(self, camera_objects):

        """
        Функция определения расстояния до ведущего на основе обработки результата детектирования объектов на изображении
        и облака точек лидара. На основе полученных bounding box происходит сопоставление их с точками лидара, используя
        результат функции calculate_points_angles_objects. В результате, из всего облака точек лидара происходит выделение
        только точек лидара, использую BB и углы из calculate_points_angles_objects.
        Далее, на основе полученной информации берется ближайшая точка, и на основе нее вычисляется расстояние до ведущего.
        Также, из всего облака точек, удаляются точки ведущего и в дальнейшем не учитвваются в обработке препятствий.

        Args:
            camera_objects = np.array()

        Returns:
            Результат:length_to_leader - расстояние до ведущего, other_points - список облака точек без ведущего
            length_to_leader = x
            other_points = list([x, y, z])

        """

        max_dist = 25
        length_to_leader = 50
        object_coord = []
        leader_info = next((x for x in camera_objects if x["name"] == "car"), None)

        other_points = list()

        camera_yaw_state_info = self.sub.get_camera_yaw_state()
        camera_yaw = camera_yaw_state_info.process_value

        # TODO : перепроверить и оптимизировать (в случае, если пробовать вариант с "закрытым" коридором)
        # if i[-1] in [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
        # выделение точек ведущего из всего облака точек
        for i in self.lidar_points:
            if leader_info is not None:
                angles_object = self._calculate_points_angles_objects(leader_info)
                if i[0] ** 2 + i[1] ** 2 <= max_dist ** 2 \
                        and (tan(np.deg2rad(-40))+tan(camera_yaw)) * i[0] <= i[1] <= (tan(np.deg2rad(40))+tan(camera_yaw)) * i[0] \
                        and ((tan(angles_object[2])+tan(camera_yaw)) * i[0]) <= i[1] <= ((tan(angles_object[0])+tan(camera_yaw)) * i[0]) \
                        and (tan(angles_object[3]) * i[0]) <= i[2] <= (tan(angles_object[1]) * i[0]):
                    object_coord.append(i)

                    if sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2) <= length_to_leader:
                        length_to_leader = sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2)
                elif i[0] ** 2 + i[1] ** 2 >= 1:
                    other_points.append([i[0], i[1], i[2]])
            elif i[0] ** 2 + i[1] ** 2 >= 1:
                other_points.append([i[0], i[1], i[2]])
                length_to_leader = None

        print(f'Расстояние до ведущего, определенное с помощью лидара: {length_to_leader}')

        return length_to_leader, other_points


    def _get_camera_lead_info(self, camera_objects, length_to_leader):
        """
        Функция определения угла отклонения ведущего относительно ведомого на основе информации с камеры и расстоянии до
        ведущего. Возвращает результат о расстояние и угле отклонения ведущего в локальных координатах
        Args:
            camera_objects = np.array()
            length_to_leader = x
        Returns:
            Информация о расстоянии и угле отклонения до лидера
            lead_results = {'length': x, 'phi': theta_new}

        """
        info_lead = next((x for x in camera_objects if x["name"] == "car"), None)
        print(info_lead)
        self.camera_leader_information = info_lead

        camera_yaw_state_info = self.sub.get_camera_yaw_state()
        camera_yaw = camera_yaw_state_info.process_value

        if bool(info_lead):
            y = (info_lead['xmin'] + info_lead['xmax']) / 2
            x = length_to_leader + 2.1
            hfov = 80
            cam_size = (640, 480)
            theta = np.arctan((2 * y - cam_size[0]) / cam_size[0] * np.tan(hfov / 2))
            # получаем ориентацию робота с gazebo и складываем с отклонением до ведущего
            yaw = self.yaw_ang
            theta_new = yaw + theta + camera_yaw
            lead_results = {'length': x, 'phi': theta_new}

            self.theta_camera_yaw = camera_yaw
            self.theta_camera_yaw += theta
            self.pub.set_camera_yaw(self.theta_camera_yaw)

        else:
            x = 0
            yaw = self.yaw_ang
            theta_new = yaw + camera_yaw
            lead_results = {'length': x, 'phi': theta_new}
            self.theta_camera_yaw = camera_yaw

        return lead_results

    def _get_delta_position(self):
        """
        Функция определения перемещения за одну итерацию. Определяется перемещение ведомого по координатам x и y за один step.
        Args:

        Returns:
            Информация о перемещении
            follower_delta_info = {'delta_x': self.delta_cx, 'delta_y': self.delta_cy}

        """
        follower_info_odom = self.sub.get_odom()
        follower_time = follower_info_odom.header.stamp.to_time()

        follower_linear_x = follower_info_odom.twist.twist.linear.x
        follower_linear_y = follower_info_odom.twist.twist.linear.y

        # ВРЕМЯ, которое используем для поиска дельта С
        self.history_time.append(follower_time)
        if len(self.history_time) > 2:
            self.delta_time = (self.history_time[1]-self.history_time[0])
            print(f"Время шага: {np.round(self.delta_time, decimals=2)}")
            self.history_time.pop(0)

        #Рассчет дельта Х
        self.history_twist_x.append(follower_linear_x)
        if len(self.history_twist_x) > 2:
            self.delta_twist_x = (self.history_twist_x[1] + self.history_twist_x[0])/2
            self.history_twist_x.pop(0)

        #Рассчет дельта Y
        self.history_twist_y.append(follower_linear_y)
        if len(self.history_twist_y) > 2:
            self.delta_twist_y = (self.history_twist_y[1] + self.history_twist_y[0])/2
            self.history_twist_y.pop(0)

        self.delta_cx = self.delta_twist_x * self.delta_time
        self.delta_cx = np.round(self.delta_cx, decimals=5)
        self.delta_cy = self.delta_twist_y * self.delta_time
        self.delta_cy = np.round(self.delta_cy, decimals=5)
        follower_delta_info = {'delta_x': self.delta_cx, 'delta_y': self.delta_cy}

        return follower_delta_info

    def _get_obs(self):
        """
        Observations среды
        """
        obs_dict = dict()
        obs_dict["LeaderCorridor_lasers"] = self.laser_values

        return np.array(obs_dict["LeaderCorridor_lasers"], dtype=np.float32)

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

            self.pub.move_base(discrete_action[0], discrete_action[1])
            rospy.sleep(self.time_for_action)
        else:
            self.pub.move_base(action[0], -action[1])
            rospy.sleep(self.time_for_action)

    def _is_done(self, leader_position, follower_position):
        """
        Функция проверки статусов выполнения задачи и нештатных ситуаций.
        Args:
            leader_position : [x, y]
            follower_position : [x, y]
        Returns:


        """
        # TODO : проверить все состояния для системы безопасности
        self.done = False
        self.is_in_box = False
        self.is_on_trace = False

        self.info = {
            "mission_status": "in_progress",
            "agent_status": "moving",
            "leader_status": "moving"
        }

        leader_status = self.sub.get_target_status()

        try:
            self.code, self.text = leader_status.status_list[-1].status, leader_status.status_list[-1].text
            print(f"Статус ведущего: {self.code}, {self.text}")
        except IndexError as e:
            print(f"Проблема получения статуса ведущего: {e}")

        # Информирование (global)
        self._trajectory_in_box()
        self._check_agent_position(self.follower_position, self.leader_position)

        if self.saving_counter % self.trajectory_saving_period == 0:
            self.leader_factual_trajectory.append(leader_position)
            self.follower_factual_trajectory.append(follower_position)
        self.saving_counter += 1

        self.step_count += 1

        # Для индикации завершения миссии ведущего
        print(f"Статус ведущего: {self.code}, {self.text}")
        if self.code == 3:
            self.info["leader_status"] = "finished"
            # self.done = True
            self.leader_finish = True

        elif self.code == 2:
            self.count_stop_leader += 1
        elif self.code == 1:
            self.count_leader_reward += 1

        else:
            return 0

        if self.roll_ang > 1 or self.roll_ang < -1 or self.pitch_ang > 1 or self.pitch_ang < -1:
            self.info["mission_status"] = "fail"
            self.info["agent_status"] = "the_robot_turned_over"
            self.crash = True
            self.done = True

            print(self.info)

            return 0

        if self.step_count > self.warm_start:
            # Система безопасности
            if self.camera_leader_information == None:
                self.info["mission_status"] = "safety system"
                self.info["leader_status"] = "None"
                self.info["agent_status"] = "moving"
                # self.done = True



            # Низкая награда
            if self.cumulated_episode_reward < self.low_reward:
                self.info["mission_status"] = "fail"
                self.info["leader_status"] = "stop"
                self.info["agent_status"] = "low_reward"
                self.crash = True
                self.done = True

                print(self.info)

                return 0

            # ведомый далеко от ведущего (global)
            if np.linalg.norm(follower_position - leader_position) > 35:
                self.info["mission_status"] = "fail"
                self.info["leader_status"] = "stop"
                self.info["agent_status"] = "too_far_from_leader"
                self.crash = True
                self.done = True

                print(self.info)

                return 0

            # ведомый далеко от ведущего
            if self.camera_lead_info['length'] > 17:
                self.info["mission_status"] = "safety system"
                self.info["leader_status"] = "stop"
                self.info["agent_status"] = "too_far_from_leader_info"
                # self.crash = True
                # self.done = True

                print(self.info)

                return 0
            # Превысило максимальное количество шагов
            if self.step_count > self.max_steps:
                self.info["mission_status"] = "finished_by_time"
                self.info["leader_status"] = "moving"
                self.info["agent_status"] = "moving"
                self.done = True

                print(self.info)

                return 0
            # Завершение следование, ведущий доехал (Local)
            if self.code == 3 and np.linalg.norm(self.goal - leader_position) < 2 \
                    and np.linalg.norm(follower_position - leader_position) < 10:
                self.info["mission_status"] = "success"
                self.info["leader_status"] = "finished"
                self.info["agent_status"] = "finished"
                self.done = True

                print(self.info)

                return 0

            # Слишком близко к последней точке истории
            if np.linalg.norm(self.leader_history_v2[-1] - [0, 0]) < 5:
                self.info["mission_status"] = "safety system"
                self.info['agent_status'] = 'too_close_from_leader_last_point'
                self.end_stop_count += 1
                if self.end_stop_count > 40:
                    self.info["mission_status"] = "safety system end"
                    self.done = True

                print(self.info)

                return 0

            # Проверка на близость к ведущему
            if self.camera_lead_info['length'] < 5.5:
                self.info["agent_status"] = "too_close_to_leader"

                print(self.info)

                return 0

            # TODO : возврат лидера в коридор
            # pus_obs = np.ones(len(self.laser_values), dtype=np.float32)
            # pus_obs[5] *= 0.65
            # pus_obs[6] *= 0.65
            # if list(obs[0:7]) == list(pus_obs)


        if self.info["leader_status"] == "moving":
            self.end_stop_count = 0
            print(self.info)
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

        # награда ведущего
        if not self.leader_finish:
            self.count_leader_steps_reward += 1

        # reward = 0.1
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
