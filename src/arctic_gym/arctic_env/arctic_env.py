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


from math import atan, tan, cos, sin
from scipy.spatial import distance

from src.continuous_grid_arctic.utils.reward_constructor import Reward
from src.arctic_gym.base_arctic_env.robot_gazebo_env import RobotGazeboEnv
from src.arctic_gym.gazebo_utils.gazebo_tracker import GazeboLeaderPositionsTracker_v2
from src.arctic_gym.gazebo_utils.gazebo_tracker import GazeboCorridor_Prev_lasers_v2
from src.continuous_grid_arctic.utils.misc import rotateVector


os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class ArcticEnv(RobotGazeboEnv):

    def __init__(self, name,
                 object_detection_endpoint,
                 time_for_action=0.2,
                 trajectory_saving_period=5,
                 leader_max_speed=1.0,
                 min_distance=6.0,
                 max_distance=25.0,
                 leader_pos_epsilon=1.25,
                 max_dev=1.5,
                 max_steps=1000,
                 low_reward=-100,
                 close_coeff=0.6,
                 use_object_detection=True
                 ):
        super(ArcticEnv, self).__init__()

        self.object_detection_endpoint = object_detection_endpoint
        self.time_for_action = time_for_action
        self.trajectory_saving_period = trajectory_saving_period
        self.leader_max_speed = leader_max_speed
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.leader_pos_epsilon = leader_pos_epsilon
        self.max_dev = max_dev
        self.warm_start = 0
        self.max_steps = max_steps
        self.low_reward = low_reward
        self.close_coeff = close_coeff
        self.use_object_detection = use_object_detection

        # Init safe zone sensor
        self.tracker_v2 = GazeboLeaderPositionsTracker_v2(host_object="arctic_robot",
                                                          sensor_name='LeaderTrackDetector',
                                                          saving_period=self.trajectory_saving_period,
                                                          corridor_width=2,
                                                          corridor_length=25)

        # Init ray sensor 1
        self.laser = GazeboCorridor_Prev_lasers_v2(host_object="arctic_robot",
                                                   sensor_name='LeaderCorridor_Prev_lasers_v2_compas',
                                                   react_to_green_zone=True,
                                                   react_to_safe_corridor=True,
                                                   react_to_obstacles=True,
                                                   lasers_count=12,
                                                   laser_length=10,
                                                   max_prev_obs=10,
                                                   pad_sectors=False)
        # Init ray sensor 2
        self.laser_aux = GazeboCorridor_Prev_lasers_v2(host_object="arctic_robot",
                                                       sensor_name='LaserPrevSensor_compas',
                                                       react_to_green_zone=False,
                                                       react_to_safe_corridor=False,
                                                       react_to_obstacles=True,
                                                       lasers_count=36,
                                                       laser_length=15,
                                                       max_prev_obs=10,
                                                       pad_sectors=False)

        # rewards
        self.reward = Reward()

    def _init_publishers(self):
        """
        Init Publishers` values
        """
        self.pub.update_corridor([])
        self.pub.set_camera_pitch(0)
        self.pub.set_camera_yaw(0)
        self.pub.update_follower_path()
        self.pub.update_target_path()

    def _init_lasers(self):
        """
        Init sensors values
        """
        self.laser.reset()
        self.laser_aux.reset()
        self.tracker_v2.reset()

    def _init_env_variables(self):
        """
        Init environment variables
        """
        # Green Zone
        self.green_zone_trajectory_points = list()
        self.leader_factual_trajectory = list()
        self.follower_factual_trajectory = list()

        self.cumulated_episode_reward = 0.0

        self.step_count = 0
        self.done = False
        self.info = {}

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

        self.end_stop_count = 0

    def reset(self):
        self._init_publishers()
        self._init_lasers()
        self._init_env_variables()

        obs, _, _, _ = self.step([0, 0])

        return obs

    def step(self, action: list):

        self._set_action(action)

        # delta x, y
        follower_delta_position = self._get_delta_position()

        # x, y, quaternion
        leader_position, follower_position, follower_orientation = self._get_positions()

        # update rviz
        self.pub.update_follower_path(*follower_position)
        self.pub.update_target_path(*leader_position)

        if self.use_object_detection:
            # JSON {object: name, xmin, ymin, width, height, score}
            ssd_camera_objects = self.get_ssd_lead_information()
            # functions with conversion to cylindrical coordinates, histogram, removal of intersections from
            # the bb machine
            length_to_leader = self.calculate_length_to_leader(ssd_camera_objects)
            # distance to leader and angle = robot orientation + angle of deviation from image center + camera
            # orientation
            leader_position_new_phi = self._get_camera_lead_info(ssd_camera_objects,
                                                                 length_to_leader,
                                                                 follower_orientation)
        else:
            leader_position_new_phi = leader_position - follower_position

        # Getting history and safe zone
        self.leader_history_v2, corridor_v2 = self.tracker_v2.scan(leader_position_new_phi,
                                                                   follower_orientation,
                                                                   follower_delta_position)

        # draw a safe zone in rviz
        cor = np.array(corridor_v2) + follower_position
        self.pub.update_corridor(cor)

        # Obtaining obstacle points and forming obs
        cur_object_points_1, cur_object_points_2 = self._get_obs_points(follower_orientation)

        # values for safe zone
        self.laser_values = self.laser.scan(follower_orientation,
                                            corridor_v2,
                                            cur_object_points_1,
                                            cur_object_points_2)

        # values for obstacles
        self.laser_aux_values = self.laser_aux.scan(follower_orientation,
                                                    corridor_v2,
                                                    cur_object_points_1,
                                                    cur_object_points_2)

        obs = self._get_obs()

        self._safe_zone(leader_position, follower_position)

        self._is_done(leader_position, follower_position, follower_orientation, leader_position_new_phi)

        reward = self._compute_reward()
        self.cumulated_episode_reward += reward

        return obs, reward, self.done, self.info

    def set_goal(self, point):
        self.goal = point

    def _safe_zone(self, leader_position, follower_position):
        first_dots_for_follower_count = int(distance.euclidean(follower_position, leader_position) * (self.leader_max_speed))

        self.leader_factual_trajectory.extend(zip(np.linspace(follower_position[0], leader_position[0], first_dots_for_follower_count),
                                                  np.linspace(follower_position[1], leader_position[1], first_dots_for_follower_count)))

    def _get_positions(self):
        """
        Obtains information about the position, direction and speed of the leader and the agent
        """
        leader_odom = self.sub.get_odom_target()
        robot_odom = self.sub.get_odom()

        leader_pos = np.array([
            leader_odom.pose.pose.position.x,
            leader_odom.pose.pose.position.y
        ])

        leader_quat = [
            leader_odom.pose.pose.orientation.x,
            leader_odom.pose.pose.orientation.y,
            leader_odom.pose.pose.orientation.z,
            leader_odom.pose.pose.orientation.w
        ]

        robot_pos = np.array([
            robot_odom.pose.pose.position.x,
            robot_odom.pose.pose.position.y
        ])

        robot_quat = [
            robot_odom.pose.pose.orientation.x,
            robot_odom.pose.pose.orientation.y,
            robot_odom.pose.pose.orientation.z,
            robot_odom.pose.pose.orientation.w
        ]

        robot_ang = np.array(tf.transformations.euler_from_quaternion(robot_quat))

        return leader_pos, robot_pos, robot_ang

    @staticmethod
    def _get_four_points(x):
        """
        Function of rounding and obtaining a neighboring point in close proximity to the received one
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
        Calculation of angles from bounding box values, calculations based on a cylindrical coordinate system
        image center - center of the coordinate system

        :param obj: object dictionary with keys name - object name; xmin, xmax, ymin, ymax - bounding box coordinates
        :param width: image width in pixels
        :param height: image height in pixels
        :param hov: horizontal camera angle
        :param fov: vertical camera angle
        :param scale: expanding the bounding box horizontally

        :return:
            Dictionary with the boundary angles of the object area vertically (phi1, phi2) and
            horizontally (theta1, theta2)
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
        Obtaining information about recognized objects from the robot's camera
        a function that sends an image from a camera to a Flask server with object detection and receives object
        detection results
        self.sub.get_from_follower_image() - accesses the ROS topic for an image from the camera

        The user needs to pass his own image to this function into his own object detection model via a post request.
        The method is used in reset and step, in those places the user needs to change the image feed or register his
        own ROS topic.

        :return:
            dictionary of objects with their boundaries in the image
        """
        image = self.sub.get_from_follower_image()
        data = image.data

        results = requests.post(self.object_detection_endpoint, data=data)

        # catch errors in receiving json
        try:
            results = json.loads(results.text)
        except json.decoder.JSONDecodeError:
            results = {}

        return results

    def get_lidar_points(self):
        lidar = self.sub.get_lidar()
        return pc2.read_points(lidar, skip_nans=False, field_names=("x", "y", "z"))

    def _get_obs_points(self, follower_orientation):
        """
        Processes a lidar point cloud to highlight obstacles. Initially, the function filters surface points using the
        CSF (Cloth Simulation Filter) method, leaving only obstacle points. Next, it normalizes them relative to the
        local position of the agent. Afterwards, the resulting list is projected onto a 2D plane, the discreteness is
        reduced and the coordinate values of the remaining points are rounded. The list is added with second
        neighboring imaginary points in the immediate vicinity to form segments, which are further verified by the
        lidar features of the neural network model.

        :param follower_orientation: the agent angle (compas)

        :return:
            Lists of projected obstacle points
            fil_ob_1 = np.array()
            fil_ob_2 = np.array()
        """
        points_list = self.get_lidar_points()
        # Filtering, getting points and passing them to the PointCloud class
        open3d_cloud = open3d.geometry.PointCloud()
        # TODO : Исправить (подумать над альтернативой + оптимизация)
        max_dist = 8
        # Cutting off the cloud of points beyond the distance from the robot
        xyz = [(x, y, z) for x, y, z in points_list if x**2 + y**2 <= max_dist**2]  # get xyz
        # print('OTHER POINTS in radius', len(xyz))

        if len(xyz) > 0:
            # Write a truncated point cloud to a pcd file
            open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
            open3d.io.write_point_cloud("test_pdal_1.pcd", open3d_cloud)
            # Inference for starting surface topography segmentation
            pc = (
                    pdal.Reader.pcd("test_pdal_1.pcd")
                    | pdal.Filter.csf(ignore="Classification[7:7]", threshold=0.6)
                    | pdal.Filter.range(limits="Classification[1:1]")  # obstacles
                    # | pdal.Filter.range(limits="Classification[2:2]")  # ground
            )
            pc.execute()
            arr_fil = pc.arrays[0]
            # print('after filtering', len(arr_fil))

            # Rounding points and writing them to a list
            list_to_arr = list()
            for i in range(len(arr_fil)):
                list_to_arr.append([np.round(arr_fil[i][0], decimals=1),
                                    np.round(arr_fil[i][1], decimals=1), 0])
            # getting a list of non-repeating points in a 2D projection
            list_fil = list()
            list_fil_2 = list()
            yaw = np.degrees(follower_orientation)[2]
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

        return fil_ob_1, fil_ob_2

    def calculate_length_to_leader(self, detected):
        """
        determines the distance to the leader based on processing the result of detecting objects in the image and the
        lidar point cloud. Based on the received bounding boxes, they are compared with lidar points using the result
        of the calculate_points_angles_objects function. As a result, only lidar points are selected from the entire
        lidar point cloud using BB and angles from calculate_points_angles_objects.
        Next, based on the information received, the nearest point is taken, and based on it, the distance to the leader
        is calculated.

        :param detected: objects obtained using the object detection neural network

        :return:
            distance to leader
        """

        camera_objects = detected.copy()

        cars = [x for x in camera_objects if x['name'] == "car"]

        max_dist = 25

        if cars == []:
            return None

        # if several cars have been identified, find the car with the largest area bounding box
        max_square = 0
        car = {}
        for one in cars:
            camera_objects.remove(one)
            square = (one['xmax'] - one['xmin']) * (one['ymax'] - one['ymin'])

            if square > max_square:
                car = one
                max_square = square

        # find the intersection of the bounding box objects with the bounding box of the car:
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

        # selecting leader points from the entire point cloud
        lidar_pts = self.get_lidar_points()

        obj_inside = []
        for obj in crossed_objects:
            object_coord = []
            for i in lidar_pts:
                angles = self._calculate_points_angles_objects(obj)

                dist = np.linalg.norm(i)

                k1_x = (tan(np.deg2rad(-40)) + tan(camera_yaw)) * i[0]
                k2_x = (tan(np.deg2rad(40)) + tan(camera_yaw)) * i[0]

                theta2_x = (tan(angles["theta2"]) + tan(camera_yaw)) * i[0]
                theta1_x = (tan(angles["theta1"]) + tan(camera_yaw)) * i[0]

                phi2_x = tan(angles["phi2"]) * i[0]
                phi1_x = tan(angles["phi1"]) * i[0]

                if dist <= max_dist and k1_x <= i[1] <= k2_x and theta2_x <= i[1] <= theta1_x and phi2_x <= i[2] <= phi1_x:
                    object_coord.append(i)

            # No object_coord points in the object area
            try:
                object_coord = np.array(object_coord)
                norms = np.linalg.norm(object_coord, axis=1)
                obj_inside.append({"name": obj["name"], "data": dict(zip(norms, object_coord))})
            except numpy.AxisError:
              pass

        # catch the moment when the lidar rays are not detected, return the distance to the leader - None
        try:
            car_data = obj_inside[0]["data"]
        except IndexError:
            return None

        # remove lidar points of other objects that intersect with the car's bounding box
        outside_car_bb = {}
        for other_data in obj_inside[1:]:
            car_keys = np.array(list(car_data.keys()))
            other_keys = np.array(list(other_data["data"].keys()))

            intersections = np.intersect1d(car_keys, other_keys)
            for inter in intersections:
                outside_car_bb[inter] = car_data.pop(inter)

        # if objects blocked all lidar rays
        if car_data == {}:
            car_data = outside_car_bb

        # using the histogram we determine the distance that occurs more often than others
        count, distance, _ = plt.hist(car_data.keys())
        idx = np.argmax(count)
        length_to_leader = distance[idx]

        return length_to_leader

    def _get_camera_lead_info(self, camera_objects, length_to_leader, follower_orientation):
        """
        determines the angle of deviation of the leader relative to the agent based on information from the camera and
        the distance to the leader

        :param camera_objects: objects obtained using the object detection neural network
        :param length_to_leader: distance to leader
        :param follower_orientation: the agent angle (compas)

        :return:
            the distance and angle of deviation of the leader in local coordinates
            lead_results = {'length': x, 'phi': theta_new}

        """
        info_lead = next((x for x in camera_objects if x["name"] == "car"), None)
        self.camera_leader_information = info_lead

        camera_yaw_state_info = self.sub.get_camera_yaw_state()
        camera_yaw = camera_yaw_state_info.process_value

        yaw = follower_orientation[2]
        if bool(info_lead) and length_to_leader is not None:
            y = (info_lead['xmin'] + info_lead['xmax']) / 2
            x = length_to_leader + 2.1
            hfov = 80
            cam_size = (640, 480)
            theta1 = atan((2 * info_lead['xmin'] - 640) / 640 * tan(hfov / 2))
            theta2 = atan((2 * info_lead['xmax'] - 640) / 640 * tan(hfov / 2))
            theta = (theta2 + theta1) / 2
            # we get the orientation of the robot from gazebo and add it with deviation to the leader
            theta_new = yaw + theta + camera_yaw
            self.theta_camera_yaw = camera_yaw
            self.theta_camera_yaw += theta
            self.pub.set_camera_yaw(self.theta_camera_yaw)

        else:
            x = 0
            theta_new = yaw + camera_yaw
            self.theta_camera_yaw = camera_yaw

        lead_x = x * cos(theta_new)
        lead_y = x * sin(theta_new)

        return np.array([lead_x, lead_y])

    def _get_delta_position(self):
        """
        Determines the movement of the robot in one step

        :return:
            delta position information
            follower_delta_info = {'delta_x': self.delta_cx, 'delta_y': self.delta_cy}

        """
        follower_info_odom = self.sub.get_odom()
        follower_time = follower_info_odom.header.stamp.to_time()

        follower_linear_x = follower_info_odom.twist.twist.linear.x
        follower_linear_y = follower_info_odom.twist.twist.linear.y

        # TIME we use to search for delta C
        self.history_time.append(follower_time)
        if len(self.history_time) > 2:
            self.delta_time = (self.history_time[1]-self.history_time[0])
            self.history_time.pop(0)

        #Calculate delta X
        self.history_twist_x.append(follower_linear_x)
        if len(self.history_twist_x) > 2:
            self.delta_twist_x = (self.history_twist_x[1] + self.history_twist_x[0])/2
            self.history_twist_x.pop(0)

        #Calculate delta Y
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
        Observations
        """
        corridor_prev_lasers_v2 = self.laser_values
        corridor_prev_lasers_v2 = np.clip(corridor_prev_lasers_v2 / self.laser.laser_length, 0, 1)

        corridor_prev_obs_lasers = self.laser_aux_values
        corridor_prev_obs_lasers = np.clip(corridor_prev_obs_lasers / self.laser_aux.laser_length, 0, 1)

        return np.concatenate((corridor_prev_lasers_v2, corridor_prev_obs_lasers), axis=1)

    def _set_action(self, action):
        self.pub.move_base(*action)
        rospy.sleep(self.time_for_action)

    def _is_done(self, leader_position, follower_position, follower_orientation, leader_position_new_phi):
        """
        checks task completion statuses and emergency situations

        :param leader_position: [x, y]
        :param follower_position: [x, y]
        :param follower_orientation: the agent angle (compas)
        :param leader_position_new_phi: vector from the agent to the leader

        """
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
        except IndexError:
            self.code = 1
            self.text = "None"

        # Informing (global)
        self._trajectory_in_box()
        self._check_agent_position(follower_position, leader_position)

        if self.saving_counter % self.trajectory_saving_period == 0:
            self.leader_factual_trajectory.append(leader_position)
            self.follower_factual_trajectory.append(follower_position)
        self.saving_counter += 1

        self.step_count += 1

        self.info["step_count"] = self.step_count

        # To indicate the completion of the leader's mission
        if self.code == 3:
            self.info["leader_status"] = "finished"

        if self.step_count > self.warm_start:
            # Safety system
            # if self.camera_leader_information == None:
            #     self.info["mission_status"] = "safety system"
            #     self.info["leader_status"] = "None"
            #     self.info["agent_status"] = "moving"
                # self.done = True

            if follower_orientation[0] > 1 or follower_orientation[0] < -1:
                self.info["mission_status"] = "fail"
                self.info["agent_status"] = "the_robot_turned_over"
                self.crash = True
                self.done = True

                print(self.info)

                return 0

            # Low reward
            if self.cumulated_episode_reward < self.low_reward:
                self.info["mission_status"] = "fail"
                self.info["leader_status"] = "stop"
                self.info["agent_status"] = "low_reward"
                self.crash = True
                self.done = True

                print(self.info)

                return 0

            # # ведомый далеко от ведущего (global)
            # if np.linalg.norm(follower_position - leader_position) > 35:
            #     self.info["mission_status"] = "fail"
            #     self.info["leader_status"] = "stop"
            #     self.info["agent_status"] = "too_far_from_leader"
            #     self.crash = True
            #     self.done = True
            #
            #     print(self.info)
            #
            #     return 0

            # the agent is far from the leader
            if np.linalg.norm(leader_position_new_phi) > 17:
                self.info["mission_status"] = "safety system"
                self.info["leader_status"] = "stop"
                self.info["agent_status"] = "too_far_from_leader_info"
                # self.crash = True
                # self.done = True

                # Зануляет скорость робота ???
                self.end_stop_count += 1
                if self.end_stop_count > 150:
                    self.info["mission_status"] = "failed by something else"
                    self.done = True

                print(self.info)

                return 0
            # Exceeded maximum number of steps
            if self.step_count > self.max_steps:
                self.info["mission_status"] = "finished_by_time"
                self.info["leader_status"] = "moving"
                self.info["agent_status"] = "moving"
                self.done = True

                print(self.info)

                return 0

            if self.code == 3 and np.linalg.norm(self.goal - leader_position) < 2 \
                    and np.linalg.norm(follower_position - leader_position) < 8.5:
                self.info["mission_status"] = "success"
                self.info["leader_status"] = "finished"
                self.info["agent_status"] = "finished"
                self.done = True

                print(self.info)

                return 0

            # Too close to the end of the story
            if np.linalg.norm(self.leader_history_v2[-1] - [0, 0]) < self.min_distance:
                self.info["mission_status"] = "safety system"
                self.info['agent_status'] = 'too_close_from_leader_last_point'
                self.end_stop_count += 1
                if self.end_stop_count > 40:
                    self.info["mission_status"] = "safety system end"
                    self.done = True

                print(self.info)

                return 0

            # Check for proximity to the leader
            if np.linalg.norm(leader_position_new_phi) < self.min_distance:
                self.info["agent_status"] = "too_close_to_leader"

                # Determining the distance to the leader is determined only by the area of the car in the frame
                # without taking into account other objects, occurs when a car is detected in the visibility area
                # but is located behind another object due to which the distance is calculated incorrectly
                self.end_stop_count += 1
                if self.end_stop_count > 40:
                    self.info["mission_status"] = "failed by obstacle in front of target"
                    self.done = True

                print(self.info)

                return 0

        if self.info["leader_status"] == "moving":
            self.end_stop_count = 0
            print(self.info)
            return 0

    def _compute_reward(self):
        reward = 0

        if self.follower_too_close:
            reward += self.reward.too_close_penalty
        else:
            if self.is_in_box and self.is_on_trace:
                reward += self.reward.reward_in_box
            elif self.is_in_box:
                reward += self.reward.reward_in_dev
            elif self.is_on_trace:
                reward += self.reward.reward_on_track
            else:
                if self.step_count > self.warm_start:
                    reward += self.reward.not_on_track_penalty

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
        """The method determines the point closest to the point from an array of points"""
        points = np.asarray(points)
        dist_2 = np.sum((points - point) ** 2, axis=1)

        if not return_id:
            return np.min(dist_2)
        else:
            return np.argmin(dist_2)
