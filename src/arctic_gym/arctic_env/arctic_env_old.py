import numpy
import open3d
import pdal
import os
import time

import tf
import json
import rospy
import numpy as np
import requests
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2


from math import atan, tan, sqrt, cos, sin
from gym.spaces import Discrete, Box
from scipy.spatial import distance

from src.continuous_grid_arctic.utils.reward_constructor import Reward
from src.arctic_gym.base_arctic_env.robot_gazebo_env import RobotGazeboEnv
from src.arctic_gym.gazebo_utils.gazebo_tracker import GazeboLeaderPositionsTracker_v2, GazeboLeaderPositionsCorridorLasers
from src.continuous_grid_arctic.utils.misc import rotateVector


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class ArcticEnv(RobotGazeboEnv):

    def __init__(self, name,
                 object_detection_endpoint,
                 discrete_action=False,
                 time_for_action=0.2,
                 trajectory_saving_period=3,
                 leader_max_speed=1.0,
                 min_distance=6.0,
                 max_distance=25.0,
                 leader_pos_epsilon=1.25,
                 max_dev=1.5,
                 max_steps=400,
                 low_reward=-200,
                 close_coeff=0.6):
        super(ArcticEnv, self).__init__()

        self.object_detection_endpoint = object_detection_endpoint

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

        self.laser = GazeboLeaderPositionsCorridorLasers(host_object="arctic_robot",
                                                         sensor_name='LeaderCorridor_lasers',
                                                         react_to_green_zone=True,
                                                         react_to_safe_corridor=True,
                                                         react_to_obstacles=False,
                                                         front_lasers_count=5,
                                                         back_lasers_count=2)

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

    def set_goal(self, point):
        self.goal = point

    def reset(self, move=True):

        self.pub.set_camera_pitch(0)
        self.pub.set_camera_yaw(0)

        self.pub.update_follower_path()
        self.pub.update_target_path()

        # print('debug')

        if move:
            # print('перемещение ведущего и ведомого в начальную позицию')
            # self.arctic_coords = [0.0, 30.0, 1.0]
            # self.target_coords = [12.0, 30.0, 1.0]

            self.arctic_coords = [0.0, -10.0, 0.3]
            self.target_coords = [10.0, -10.0, 0.7]

            # self.arctic_coords = [40.0, 0.0, 1.0]
            # self.target_coords = [47.0, 0.0, 1.0]

            self.pub.teleport(model="arctic_model", point=self.arctic_coords, quaternion=[0, 0, 0, 1])
            self.pub.teleport(model="target_robot", point=self.target_coords, quaternion=[0, 0, 0, 1])

        # self._reset_sim()
        self._init_env_variables()
        self.follower_delta_position = self._get_delta_position()

        """для reset image data как в step()"""
        self.leader_position, self.follower_position, self.follower_orientation = self._get_positions()
        self.roll_ang, self.pitch_ang, self.yaw_ang = tf.transformations.euler_from_quaternion(
            self.follower_orientation)

        # Вызов основных функций
        self.ssd_camera_objects = self.get_ssd_lead_information()
        # self.get_lidar_points()
        self.length_to_leader, self.other_points = self.calculate_length_to_leader(self.ssd_camera_objects)
        self.camera_lead_info = self._get_camera_lead_info(self.ssd_camera_objects, self.length_to_leader)
        # Сопоставление и получение координат ведущего на основе информации о расстояние и угле отклонения
        self.leader_position_new_phi = self._get_xy_lead_from_length_phi(self.camera_lead_info)


        # Получение истории и корридора
        self.leader_history_v2, self.corridor_v2 = self.tracker_v2.scan(self.leader_position_new_phi,
                                                                        self.follower_position,
                                                                        self.follower_delta_position)
        # Получение точек препятствий и формирование obs
        self.cur_object_points_1, self.cur_object_points_2 = self._get_obs_points(self.other_points)

        print("Object points 1: {}".format(self.cur_object_points_1))
        print("Object points 2: {}".format(self.cur_object_points_2))

        self.laser_values = self.laser.scan([0.0, 0.0], self.follower_orientation, self.leader_history_v2,
                                            self.corridor_v2, self.cur_object_points_1, self.cur_object_points_2)

        obs = self._get_obs()
        """"""
        self._safe_zone()
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

        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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
        self.ssd_camera_objects = self.get_ssd_lead_information()
        # self.get_lidar_points()

        self.length_to_leader, self.other_points = self.calculate_length_to_leader(self.ssd_camera_objects)
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

        # if self.done:
        #     # print(f"Количество шагов: {self.step_count}, шагов вне зоны следования: {self.steps_out_box}")
        #     # print(f"Общая награда: {np.round(self.cumulated_episode_reward, decimals=2)}")
        #     # self._test_sys()

        return obs, reward, self.done, self.info

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
    def _calculate_points_angles_objects(obj: dict,
                                        width: int = 640,
                                        height: int = 480,
                                        hov: float = 80.0,
                                        fov: float = 64.0,
                                        scale: int = 10) -> dict:
        """
        Вычисление углов по значениям bounding box, вычисления основано на цилиндрической системе координат,
        центр изображения - центр системы координат

        :param obj: словарь объекта с ключами name - имя объекта; xmin, xmax, ymin, ymax - координаты bounding box
        :param width: ширина изображения в пикселях
        :param height: высота изображения в пикселях
        :param hov: горизонтальный угол обзора камеры
        :param fov: вертикальный угол обзора камеры
        :param scale: расширение границ bounding box по горизонтали

        :return:
            Словарь с граничными углами области объекта по вертикали (phi1, phi2) и горизонтали (theta1, theta2)
        """

        xmin = obj['xmin']
        xmax = obj['xmax']
        ymin = obj['ymin']
        ymax = obj['ymax']

        xmin -= scale
        xmax += scale

        theta1 = atan((2 * xmin - width) / width * tan(hov / 2))
        theta2 = atan((2 * xmax - width) / width * tan(hov / 2))

        phi1 = atan(-((2 * ymin - height) / height) * tan(fov / 2))
        phi2 = atan(-((2 * ymax - height) / height) * tan(fov / 2))

        return {
            "theta1": theta1,
            "theta2": theta2,
            "phi1": phi1,
            "phi2": phi2
        }

    def get_ssd_lead_information(self) -> dict:
        """
        Получение информации о распознанных объектах с камеры робота
        функция отправляющая изображение с камеры на Flask сервер с object detection и получающая результаты
        детектирования объектов
        self.sub.get_from_follower_image() - обращается к топику ROS за изображением с камеры

        Пользователю необходимо передавать в данную функцию собственной изображение в собственную модель
        детектирования объектов посредством post запроса.
        Метод используется в reset и step, в тех местах пользователю необходимо изменить подачу изображения или прописать
        собственный топик ROS.
        :return:
            словарь объектов с их границами на изображении
        """
        image = self.sub.get_from_follower_image()
        data = image.data

        results = requests.post(self.object_detection_endpoint, data=data)

        # ловим ошибки получения json
        try:
            results = json.loads(results.text)
        except json.decoder.JSONDecodeError:
            results = {}

        return results

    def get_lidar_points(self):
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
        # print('OTHER POINTS in radius', len(xyz))

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
            # print('after filtering', len(arr_fil))

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

        # print("TIME FILTERING", time.time() - t1)
        # self._test_lid_filters(len(xyz), len(arr_fil), t2_end, time.time() - t1)

        return fil_ob_1, fil_ob_2

    def calculate_length_to_leader(self, detected):

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
        start = time.time()

        camera_objects = detected.copy()

        cars = [x for x in camera_objects if x['name'] == "car"]

        max_dist = 25
        length_to_leader = 50
        other_points = []

        # если нет машины передаем все точки лидара как препятствия
        if cars == []:
            self.get_lidar_points()
            length_to_leader = None
            for i in self.lidar_points:
                dist = np.linalg.norm(i[:3])
                if dist >= 1:
                    other_points.append(i[:3])

            return length_to_leader, other_points

        # если определилось несколько машин, находим машину с наибольшей площадью bounding box
        max_square = 0
        car = {}
        for one in cars:
            camera_objects.remove(one)
            square = (one['xmax'] - one['xmin']) * (one['ymax'] - one['ymin'])

            if square > max_square:
                car = one
                max_square = square

        # находим пересечение bounding box объектов с bounding box машины:
        crossed_objects = [car, ]
        for obj in camera_objects:
            x1_c = car['xmin']
            x2_c = car['xmax']
            y1_c = car['ymin']
            y2_c = car['ymax']

            x1_o = obj['xmin']
            x2_o = obj['xmax']
            y1_o = obj['ymin']
            y2_o = obj['ymax']

            if (x1_c < x2_o and x2_c > x1_o) and (y1_c < y2_o and y2_c > y1_o):
                crossed_objects.append(obj)

        camera_yaw = self.sub.get_camera_yaw_state().process_value

        # выделение точек ведущего из всего облака точек
        self.get_lidar_points()
        lidar_pts = list(self.lidar_points).copy()

        obj_inside = []
        for obj in crossed_objects:
            object_coord = []
            for i in lidar_pts:
                angles = self._calculate_points_angles_objects(obj)

                dist = np.linalg.norm(i[:3])

                k1_x = (tan(np.deg2rad(-40)) + tan(camera_yaw)) * i[0]
                k2_x = (tan(np.deg2rad(40)) + tan(camera_yaw)) * i[0]

                theta2_x = (tan(angles["theta2"]) + tan(camera_yaw)) * i[0]
                theta1_x = (tan(angles["theta1"]) + tan(camera_yaw)) * i[0]

                phi2_x = tan(angles["phi2"]) * i[0]
                phi1_x = tan(angles["phi1"]) * i[0]

                if dist <= max_dist and k1_x <= i[1] <= k2_x and theta2_x <= i[1] <= theta1_x and phi2_x <= i[2] <= phi1_x:
                    object_coord.append(i[:3])

                    # if dist <= length_to_leader:
                    #     length_to_leader = dist
                elif dist >= 1 and obj["name"]:
                    other_points.append(i[:3])

            # Отсутствие точек object_coord в области объекта
            try:
                object_coord = np.array(object_coord)
                norms = np.linalg.norm(object_coord, axis=1)
                obj_inside.append({"name": obj["name"], "data": dict(zip(norms, object_coord))})
            except numpy.AxisError:
              pass

        # ловим момент когда не выделились лучи лидара, возвращаем расстояние до ведущего - None
        try:
            car_data = obj_inside[0]["data"]
        except IndexError:
            length_to_leader = None

            return length_to_leader, other_points

        # удаляем точки лидара других объектов, которые пересекаются с bounding box машины
        outside_car_bb = {}
        for other_data in obj_inside[1:]:
            car_keys = np.array(list(car_data.keys()))
            other_keys = np.array(list(other_data["data"].keys()))

            intersections = np.intersect1d(car_keys, other_keys)
            for inter in intersections:
                outside_car_bb[inter] = car_data.pop(inter)

        # если объекты перекрыли все лучи лидара
        if car_data == {}:
            car_data = outside_car_bb

        # по гистограмме определяем расстояние, которое встречается чаще остальных
        count, distance, _ = plt.hist(car_data.keys())
        idx = np.argmax(count)
        length_to_leader = distance[idx]

        print("DISTANCE: {}".format(length_to_leader))

        print("TIME: {}".format(time.time() - start))

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
        # print(info_lead)
        self.camera_leader_information = info_lead

        camera_yaw_state_info = self.sub.get_camera_yaw_state()
        camera_yaw = camera_yaw_state_info.process_value

        if bool(info_lead) and length_to_leader is not None:
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
            # print(f"Время шага: {np.round(self.delta_time, decimals=2)}")
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

        print("STEP: {}".format(self.step_count))

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
            # print(f"Статус ведущего: {self.code}, {self.text}")
        except IndexError as e:
            self.code = 1
            self.text = "None"
        #     print(f"Проблема получения статуса ведущего: {e}")

        # Информирование (global)
        self._trajectory_in_box()
        self._check_agent_position(self.follower_position, self.leader_position)

        if self.saving_counter % self.trajectory_saving_period == 0:
            self.leader_factual_trajectory.append(leader_position)
            self.follower_factual_trajectory.append(follower_position)
        self.saving_counter += 1

        self.step_count += 1

        # Для индикации завершения миссии ведущего
        # print(f"Статус ведущего: {self.code}, {self.text}")
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

                # Зануляет скорость робота ???
                self.end_stop_count += 1
                if self.end_stop_count > 40:
                    self.info["mission_status"] = "failed by something else"
                    self.done = True

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

            print(self.code)
            print(np.linalg.norm(self.goal - leader_position))
            print(np.linalg.norm(follower_position - leader_position))

            if self.code == 3 and np.linalg.norm(self.goal - leader_position) < 2 \
                    and np.linalg.norm(follower_position - leader_position) < 10:
                self.info["mission_status"] = "success"
                self.info["leader_status"] = "finished"
                self.info["agent_status"] = "finished"
                self.done = True

                print(self.info)

                return 0

            # Слишком близко к последней точке истории
            if np.linalg.norm(self.leader_history_v2[-1] - [0, 0]) < self.min_distance:
                self.info["mission_status"] = "safety system"
                self.info['agent_status'] = 'too_close_from_leader_last_point'
                self.end_stop_count += 1
                if self.end_stop_count > 40:
                    self.info["mission_status"] = "safety system end"
                    self.done = True

                print(self.info)

                return 0

            # Проверка на близость к ведущему
            if self.camera_lead_info['length'] < self.min_distance:
                self.info["agent_status"] = "too_close_to_leader"

                # Определение расстояния до ведущего определяется только по области машины в кадре
                # без учета других объектов, возникает когда машина детектируется в области видимости
                # но располагается за другим объектом из-за чего расстояние рассчитывается неправильно
                self.end_stop_count += 1
                if self.end_stop_count > 40:
                    self.info["mission_status"] = "failed by obstacle in front of target"
                    self.done = True

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