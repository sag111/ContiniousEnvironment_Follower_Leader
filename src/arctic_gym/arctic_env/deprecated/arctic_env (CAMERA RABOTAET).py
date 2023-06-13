import os
import rospy
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from random import choice
from scipy.spatial import distance
from src.arctic_gym.utils.reward_constructor import Reward
from src.arctic_gym.base_arctic_env.robot_gazebo_env import RobotGazeboEnv
from src.arctic_gym.gazebo_utils.gazebo_tracker import GazeboLeaderPositionsTracker, GazeboLeaderPositionsTrackerRadar
# from arctic_gym.utils.CustomFormatter import logger
from src.arctic_gym.utils.CustomFormatter import ColoredFormatter

from gym.spaces import Discrete, Box

import cv2
import io
from PIL import Image
import math
import logging
import sys
import tf
import pandas as pd
import json
import requests
from sklearn.metrics import r2_score


# log, formatter = logger(name='arctic_env', level='INFO')
log = logging.getLogger('arctic_env')
formatter = ColoredFormatter()

log.setLevel(logging.DEBUG)
handler_stream = logging.StreamHandler(stream=sys.stdout)
handler_stream.setFormatter(formatter)
handler_stream.setLevel(logging.DEBUG)
log.addHandler(handler_stream)

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

        self.targets = [[50.0, -40.0]]

    def reset(self):
        self._reset_sim()
        self._init_env_variables()

        self.camera_results = self._get_follower_camera_info()
        self.follower_length = self._get_follower_length(results=self.camera_results)
        self.follower_delta_position = self._get_delta_position()
        self.lead_info = self._get_lead_position_info(results=self.camera_results)

        # self.follower_delta_info = self._get_delta_pose_from_twist_controller()
        # log.info(self.follower_delta_position)

        """для reset image data как в step()"""

        self.leader_position, self.follower_position, self.follower_orientation = self._get_positions()
        # self.leader_history = self.tracker.scan(self.leader_position, self.follower_position)

        self.leader_position_new = self._get_leader_local_pose_info(self.leader_position, self.follower_position,
                                                                    self.follower_orientation)

        self.lead_length_phi_from_xy = self._get_length_phi_lead_from_xy(self.leader_position_new)

        # self.leader_position_new_phi = self._get_xy_lead_from_length_phi(self.lead_length_phi_from_xy)

        self.camera_lead_info = self._get_camera_lead_info()

        self.leader_position_new_phi = self._get_xy_lead_from_length_phi(self.camera_lead_info)

        self.leader_history = self.tracker.scan(self.leader_position_new_phi, self.follower_position,
                                                self.follower_delta_position)

        self.radar_values = self.radar.scan([0,0], self.follower_orientation, self.leader_history)

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

        self.history_time = list()
        self.delta_time = 0

        self.history_time_twist = list()
        self.delta_time_twist = 0

        self.history_linear = list()
        self.delta_linear = 0

        self.history_angular = list()
        self.delta_angular = 0

        self.history_teta_direction = 0


        ############################
        self.history_twist_x = list()
        self.delta_twist_x = 0

        self.history_twist_y = list()
        self.delta_twist_y = 0

        self.follower_hist_x = list()
        self.follower_hist_y = list()
        self.delta_x_pose = 0
        self.delta_y_pose = 0

    def step(self, action):
        log_linear = formatter.colored_logs(round(float(action[0]), 2), 'obs')
        log_angular = formatter.colored_logs(round(float(action[1]), 2), 'obs')
        log.debug(f'Actions: linear - {log_linear}, angular - {log_angular}')
        self._set_action(action)
        self.camera_results = self._get_follower_camera_info()
        self.follower_length = self._get_follower_length(results=self.camera_results)
        self.follower_delta_position = self._get_delta_position()
        self.lead_info = self._get_lead_position_info(results=self.camera_results)

        # self.follower_delta_info = self._get_delta_pose_from_twist_controller()
        # log.info(self.follower_delta_position)
        """
        Радар по позициям ведущего и ведомого
        """
        self.leader_position, self.follower_position, self.follower_orientation = self._get_positions()
        log.info(self.leader_position)
        log.info(self.follower_position)

        self.leader_position_new = self._get_leader_local_pose_info(self.leader_position, self.follower_position,
                                                                    self.follower_orientation)

        self.lead_length_phi_from_xy = self._get_length_phi_lead_from_xy(self.leader_position_new)

        # self.leader_position_new_phi = self._get_xy_lead_from_length_phi(self.lead_length_phi_from_xy)

        self.camera_lead_info = self._get_camera_lead_info()

        self.leader_position_new_phi = self._get_xy_lead_from_length_phi(self.camera_lead_info)
        log.info('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        log.info(self.leader_position_new)
        log.info(self.leader_position_new_phi)
        log.info(r2_score(self.leader_position_new, self.leader_position_new_phi))
        log.info('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')


        self.leader_history = self.tracker.scan(self.leader_position_new_phi, self.follower_position,
                                                self.follower_delta_position)
        log.info(self.leader_history)

        # self.radar_values = self.radar.scan(self.follower_position, self.follower_orientation, self.leader_history)
        self.radar_values = self.radar.scan([0,0], self.follower_orientation, self.leader_history)


        obs = self._get_obs()
        """"""

        self._is_done(self.leader_position, self.follower_position)

        log_obs = formatter.colored_logs(list(map(lambda x: round(x, 2), obs)), 'obs')
        log.debug(f'Observations: {log_obs}')

        reward = self._compute_reward()
        self.cumulated_episode_reward += reward

        log.info(self.cumulated_episode_reward)

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
            rew_final = self.cumulated_episode_reward
            path_tab = '~/Desktop/arctic_rob_2.xlsx'
            table = pd.read_excel(path_tab, index_col=False)
            new_data = {'metod5': rew_final, 'Step_count_5': self.step_count, 'steps_out_5': self.steps_out_box}
            new_table = table.append(new_data, ignore_index=True)
            new_table.to_excel(path_tab, index=False)

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

    def _get_follower_camera_info(self):
        image = self.gz_subscribers.get_from_follower_image()
        data = image.data
        # print(data)
        # results = requests.post('http://localhost:6789/detection', data=data, timeout=5.0)
        # results = json.loads(results.text[12:-2])
        # print(results)
        image1 = np.array(Image.open(io.BytesIO(data)))
        # print(image1)
        img2gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        ret, maska = cv2.threshold(img2gray, 110, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(maska)
        cv2.waitKey(1)
        # cv2.imshow('mask', mask_inv)

        # get contours
        result = image1.copy()
        x = 10001
        w = 0
        y = 0
        h = 0
        contours = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)

        length = 30
        name = 'leader'
        xmax = x + w
        xmin = x
        ymax = y + h
        ymin = y

        results = {'length': length, 'name': name, 'score': 1,
                   'xmax': xmax, 'xmin': xmin, 'ymax': ymax, 'ymin': ymin}
        # log.info(results)

        return results

    def _get_follower_length(self, results):
        self.high_lead_image = results['ymax'] - results['ymin']
        self.length_to_leader = 0.00016 * self.high_lead_image ** 2 - 0.10420 * self.high_lead_image + 20.70245
        # log.info(self.length_to_leader)
        return self.length_to_leader

    def _get_leader_local_pose_info(self, leader_position, follower_position, follower_orientation_list):

        local_lead_pose = leader_position - follower_position
        # print("LEADER POSITION : ", leader_position)
        # print("LEADER POSITION : ", follower_position)
        # print("LOCAL POSITION : ", local_lead_pose)
        return local_lead_pose

    def _get_delta_pose_from_twist_controller(self):

        controller_twist = self.gz_subscribers.get_twist_controller()
        linear_velocity = controller_twist.twist.linear.x
        angular_velocity = controller_twist.twist.angular.z
        time = controller_twist.header.stamp.to_time()
        # print("__________________________")
        # log.info(linear_velocity)
        # print("__________________________")
        # log.info(angular_velocity)
        # print("__________________________")
        # log.info(time)
        # print("__________________________")

        self.history_time_twist.append(time)
        if len(self.history_time_twist) > 1:
            self.delta_time_twist = (self.history_time_twist[1] - self.history_time_twist[0])
            print("DELTA TIME : ", self.delta_time_twist)
            self.history_time_twist.pop(0)
        #
        # # рассчет делта омега
        # self.history_angular.append(angular_velocity)
        # if len(self.history_angular) > 2:
        #     self.delta_angular = (self.history_angular[1] + self.history_angular[0])/2
        #     self.history_angular.pop(0)
        #
        # #Рассчет дельта линеар
        # self.history_linear.append(linear_velocity)
        # if len(self.history_linear) > 2:
        #     self.delta_linear = (self.history_linear[1] + self.history_linear[0])/2
        #     self.history_linear.pop(0)


        teta = angular_velocity * self.delta_time_twist
        self.history_teta_direction += teta

        try:
            delta_cx = -(linear_velocity/angular_velocity) * math.sin(self.history_teta_direction) + (linear_velocity/angular_velocity) * \
                            math.sin(angular_velocity*self.delta_time_twist + self.history_teta_direction)
        except ZeroDivisionError:
            # delta_cx = 0
            print("NUULLLLLL")
            angular_velocity = 0.000000001
            delta_cx = -(linear_velocity / angular_velocity) * math.sin(self.history_teta_direction) + (
                        linear_velocity / angular_velocity) * \
                       math.sin(angular_velocity * self.delta_time_twist + self.history_teta_direction)
        # log.info(delta_cx)


        try:
            delta_cy = (linear_velocity/angular_velocity) * math.cos(self.history_teta_direction) - (linear_velocity/angular_velocity) * \
                            math.cos(angular_velocity*self.delta_time_twist + self.history_teta_direction)
        except ZeroDivisionError:
            delta_cy = 0
            print("NUULLLLLL")
            angular_velocity = 0.000000001
            delta_cy = (linear_velocity / angular_velocity) * math.cos(self.history_teta_direction) - (
                        linear_velocity / angular_velocity) * \
                            math.cos(angular_velocity * self.delta_time_twist + self.history_teta_direction)
        # log.info(delta_cy)
        # print("__________________________")
        # print("__________________________")
        # self.history_teta_direction += teta
        # teta_degre = np.degrees(self.history_teta_direction )
        # log.info(teta_degre)
        # _, _, yaw = tf.transformations.euler_from_quaternion(self.follower_orientation)
        # teta_follower_degree = np.degrees(yaw)
        # log.info(teta_follower_degree)
        # delta_cx = self.delta_linear * math.cos(teta)
        # delta_cx = np.round(delta_cx, decimals=2)
        # delta_cy = self.delta_linear * math.sin(teta)
        # delta_cy = np.round(delta_cy, decimals=2)

        # print("______________TETA____________")
        # log.info(np.degrees(self.history_teta_direction))
        # print("__________DELTA_CX________________")
        # log.info(delta_cx)
        # print("__________DELTA_CY________________")
        # log.info(delta_cy)
        # print("__________________________")

        delta_cx = np.round(delta_cx, decimals=2)
        delta_cy = np.round(delta_cy, decimals=2)

        follower_delta_twist = {'delta_x': delta_cx, 'delta_y': delta_cy, 'teta': teta}

        return follower_delta_twist


    def _get_length_phi_lead_from_xy(self, leader_pose):
        lead_x = leader_pose[0]
        lead_y = leader_pose[1]

        length = math.sqrt(lead_x**2 + lead_y**2)
        # length = np.round(length, decimals=2)
        phi = math.atan(lead_y/lead_x)
        # phi = np.round(phi, decimals=2)
        log.info("______________________________")
        # log.info(lead_x)
        # log.info(lead_y)
        log.info(length)
        log.info(phi)
        log.info(np.degrees(phi))
        log.info("______________________________")
        results = {'length': length, 'phi': phi}
        return results

    def _get_xy_lead_from_length_phi(self, length_phi):
        length = length_phi['length']
        phi = length_phi['phi']

        lead_x = length * math.cos(phi)
        # lead_x = np.round(lead_x, decimals=2)
        lead_y = length * math.sin(phi)
        # lead_y = np.round(lead_y, decimals=2)
        # log.info('+++++++++++++++++++++++++++++++++')
        # log.info(lead_x)
        # log.info(lead_y)
        # log.info('+++++++++++++++++++++++++++++++++')

        # results = {'lead_x': lead_x, 'lead_y': lead_y}
        results = np.array([np.round(lead_x, decimals=2), np.round(lead_y, decimals=2)])
        return results

    def _get_camera_lead_info(self):
        image = self.gz_subscribers.get_from_follower_image()
        data = image.data
        # print(data)
        results = requests.post('http://localhost:6789/detection', data=data, timeout=5.0)
        # results = json.loads(results.text[12:-2])
        # results = next(x for x in results if x["name"] == "car")
        results = json.loads(results.text)
        # log.info(results)
        info_lead = next((x for x in results if x["name"] == "car"), None)
        # print(next(x for x in results if x["name"] == "car"), None)
        # print(results)
        # log.info("++++++++!!!!!!!!!!!!!++++++++++")
        log.info(info_lead)
        if bool(info_lead):
            y = (info_lead['xmin'] + info_lead['xmax']) / 2
            z = (info_lead['ymin'] + info_lead['ymax']) / 2
            x = info_lead['length']

            hfov = 80
            vfov = 64
            cam_size = (640, 480)

            theta = np.arctan((2 * y - cam_size[0]) / cam_size[0] * np.tan(hfov / 2))
            # theta = np.arctan((2 * y - self.conf.size[0]) / self.conf.size[0] * np.tan(self.conf.hfov / 2))
            log.info("..........................................................")
            log.info(x)
            # log.info(theta)
            _, _, yaw = tf.transformations.euler_from_quaternion(self.follower_orientation)
            theta_new = yaw + theta
            log.info(theta_new)
            log.info(np.degrees(theta_new))
            log.info("..........................................................")
            lead_results = {'length': x, 'phi': theta_new}
        else:
            x = 0
            _, _, yaw = tf.transformations.euler_from_quaternion(self.follower_orientation)
            theta_new = yaw
            lead_results = {'length': x, 'phi': theta_new}

        return lead_results

    def _get_lead_position_info(self, results, cam_size = (640, 480)):
        w1 = results['xmax'] - results['xmin']
        h1 = results['ymax'] - results['ymin']
        x1 = results['xmin']
        y1 = results['ymin']
        x_centre = int(x1 + w1 / 2)
        y_centre = int(y1 + h1 / 2)

        y = x_centre
        z = y_centre
        x = self._get_follower_length(results=self.camera_results)
        # target_odom = self.gz_subscribers.get_odom_target()
        # arctic_odom = self.gz_subscribers.get_odom()
        # x = math.sqrt((np.round(target_odom.pose.pose.position.x, decimals=2) -
        #                np.round(arctic_odom.pose.pose.position.x, decimals=2))**2 +
        #               (np.round(target_odom.pose.pose.position.y, decimals=2) -
        #                np.round(arctic_odom.pose.pose.position.y, decimals=2))**2)
        x = np.round(x, decimals=2)
        log.info(x)
        hfov = 80
        vfov = 64

        theta = np.arctan((2 * y - cam_size[0]) / cam_size[0] * np.tan(hfov / 2))
        # print("DEC THETA IMAGE : ", theta)
        phi = np.arctan((2 * z - cam_size[1]) / cam_size[1] * np.tan(vfov / 2))
        # print("DEC PHI IMAGE : ", phi)
        length = np.sqrt(1 + math.sin(theta) ** 2 + math.sin(phi) ** 2)

        dec_x = 1 / length * x
        # print("DEC X : ", dec_x)
        dec_y = math.sin(theta) / length * x
        # print("DEC Y : ", dec_y)
        dec_z = math.sin(phi) / length * x
        # print("DEC Z : ", dec_z)

        lead_pose = np.array([np.round(dec_x, decimals=2), np.round(dec_y, decimals=2)])


        lead_info = {'theta': theta, 'phi': phi, 'dec_x': dec_x, 'dec_y': dec_y, 'dec_z': dec_z, 'lead_pose': lead_pose}

        return lead_info


    def _get_delta_position(self):

        follower_info_odom = self.gz_subscribers.get_odom()
        follower_time = follower_info_odom.header.stamp.to_time()
        # follower_linear = follower_info_odom.twist.twist.linear.x
        # follower_angular = follower_info_odom.twist.twist.angular.z
        # follower_velocity_info = {'time' : follower_time, 'linear' : follower_linear, 'angular' : follower_angular}

        follower_linear_x = follower_info_odom.twist.twist.linear.x
        follower_linear_y = follower_info_odom.twist.twist.linear.y
        # follower_velocity_info = {'time': follower_time, 'linear_x :': follower_linear_x, 'linear_y': follower_linear_y}

        # print(dir(follower_time))
        # print(follower_time.to_time())

        # ВРЕМЯ которое используем для поиска дельта С
        self.history_time.append(follower_time)
        if len(self.history_time) > 2:
            self.delta_time = (self.history_time[1]-self.history_time[0])
            print("DELTA TIME : ", self.delta_time)
            # self.delta_time_list.append(delta_time)
            self.history_time.pop(0)

        #Рассчет дельта Х
        self.history_twist_x.append(follower_linear_x)
        if len(self.history_twist_x) > 2:
            self.delta_twist_x = (self.history_twist_x[1] + self.history_twist_x[0])/2
            # self.delta_twist_x = np.round(self.delta_twist_x, decimals=2)
            # self.delta_twist_x_list.append(delta_twist_x)
            self.history_twist_x.pop(0)

        #Рассчет дельта Y
        self.history_twist_y.append(follower_linear_y)
        if len(self.history_twist_y) > 2:
            self.delta_twist_y = (self.history_twist_y[1] + self.history_twist_y[0])/2
            # self.delta_twist_y = np.round(self.delta_twist_y, decimals=2)
            # self.delta_twist_y_list.append(delta_twist_y)
            self.history_twist_y.pop(0)

        self.delta_cx = self.delta_twist_x * self.delta_time
        self.delta_cx = np.round(self.delta_cx, decimals=2)
        self.delta_cy = self.delta_twist_y * self.delta_time
        self.delta_cy = np.round(self.delta_cy, decimals=2)
        follower_delta_info = {'delta_x': self.delta_cx, 'delta_y': self.delta_cy}
        # return follower_velocity_info
        return follower_delta_info

    def _get_leader_delta_positions(self, lead_info, delta):

        leader_position_x = lead_info['dec_x']
        leader_position_y = lead_info['dec_y']

        delta_cx = delta['delta_x']
        delta_cy = delta['delta_y']
        log.info(delta_cx)
        log.info(delta_cy)


        leader_position_x -= delta_cx
        # leader_position_x = np.round(leader_position_x, decimals=2)
        leader_position_y -= delta_cy
        # leader_position_y = np.round(leader_position_y, decimals=2)

        leader_position = np.array([np.round(leader_position_x, decimals=2),
                                    np.round(leader_position_y, decimals=2)])
        return leader_position

    def _get_follower_delta_positions(self, delta):
        delta_cx = delta['delta_x']
        delta_cy = delta['delta_y']

        follower_x = self.follower_position_delta[0]
        follower_y = self.follower_position_delta[1]

        follower_x += delta_cx
        # follower_x = np.round(follower_x, decimals=2)
        follower_y += delta_cy
        # follower_y = np.round(follower_y, decimals=2)

        # self.follower_position_delta = np.array([follower_x, follower_y])
        follower_pose = np.array([follower_x, follower_y])
        return follower_pose

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
