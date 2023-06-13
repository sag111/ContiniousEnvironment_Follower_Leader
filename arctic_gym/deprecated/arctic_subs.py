#!/usr/bin/env python

import sys
import rospy
import logging

from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from actionlib_msgs.msg import GoalStatusArray
# TODO: from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CompressedImage
from src.arctic_gym.utils.CustomFormatter import ColoredFormatter

from geometry_msgs.msg import TwistStamped
from control_msgs.msg import JointControllerState

import sensor_msgs.point_cloud2 as pc2



log = logging.getLogger('gazebo_sub')
log.setLevel(logging.DEBUG)

handler_stream = logging.StreamHandler(stream=sys.stdout)
handler_stream.setFormatter(ColoredFormatter())
handler_stream.setLevel(logging.INFO)

log.addHandler(handler_stream)


class ArcticSubscribers:
    """
    Subscribers топиков ROS для арктики
    """
    def __init__(self):
        self.check_all_subscribers_ready()

        # Координаты ведомого робота
        rospy.Subscriber("/default_robot/gazebo_ground_truth_odom", Odometry, self._odom_callback)
        # Координаты ведущего робота
        rospy.Subscriber("/target_robot/target_ground_truth_odom", Odometry, self._odom_target_callback)
        # Показания лидара
        rospy.Subscriber("/default_robot/velodyne_points2", PointCloud2, self._lidar_callback)
        # Путь ведущего робота до точки
        rospy.Subscriber("/target_robot/move_base/TebLocalPlannerROS/global_plan", Path, self._target_path_callback)
        # Статус выполения команды ведущего робота
        rospy.Subscriber("/target_robot/move_base/status", GoalStatusArray, self._target_status_callback)

        # TODO : обычная камера для lidar
        # Получение изображения с камеры ведомого
        # rospy.Subscriber("/default_robot/camera1/image_raw/compressed", CompressedImage, self._follower_image_stat_callback)


        # TODO: вращательная камера для лидера рада и тд
        rospy.Subscriber("/default_robot/rotating_camera/image_raw/compressed", CompressedImage,
                            self._follower_image_callback)

        # rospy.Subscriber("/default_robot/camera1/image_raw/compressed", CompressedImage,
        #                  self._follower_image_callback)



        # TODO : получение ориентации с камеры
        rospy.Subscriber("/default_robot/camera_yaw_controller/state", JointControllerState,
                         self._follower_camera_yaw_state)
        rospy.Subscriber("/default_robot/camera_pitch_controller/state", JointControllerState,
                         self._follower_camera_pitch_state)


        rospy.Subscriber("/default_robot/mobile_base_controller/cmd_vel_out", TwistStamped, self._conroller_twist)

        # # TODO :lidar Kirill
        # rospy.Subscriber("/default_robot/velodyne_points2", PointCloud2, self._lidar_callback)

    def get_odom(self):
        """
        Получение Odomtetry (позиции, ориентации, скорости) ведомого робота
        """
        return self.odom

    def get_odom_target(self):
        """
        Получение Odometry (позиции, ориентации, скорости) ведущего робота
        """
        return self.odom_target

    def get_lidar(self):
        """
        Получение PointCloud2 значений лидара, ведомого робота
        """
        # return self.lidar
        return self.lidar_points

    def get_target_path(self):
        """
        Получение Path пути планировщика ведущего робота
        """
        return self.target_path

    def get_target_status(self):
        """
        Получение GoalStatusArray статуса выполнения команды ведущего робота
        """
        return self.target_status

    # TODO: get_image_from_follower()
    def get_from_follower_image(self):
        """
        Получение изображения с камеры ведомого
        """
        return self.follower_image

    def get_twist_controller(self):
        return self.follower_controller_twist

    def get_camera_yaw_state(self):
        return self.camera_yaw_state

    def get_camera_pitch_state(self):
        return self.camera_pitch_state

    # def get_lidar_info(self):
    #     return self.lidar

    # def __lidar_callback(self, scan):
    #     self.data_lidar = scan
    #     self.gen = pc2.read_points(self.data_lidar, skip_nans=False, field_names=("x", "y", "z"))

    ############################

    def _odom_callback(self, data):
        self.odom = data

    def _odom_target_callback(self, data):
        self.odom_target = data

    def _lidar_callback(self, data):
        self.lidar = data
        self.lidar_points = pc2.read_points(self.lidar, skip_nans=False, field_names=("x", "y", "z", "ring"))

    def _target_path_callback(self, data):
        self.target_path = data

    def _target_status_callback(self, data):
        self.target_status = data

    # TODO _image_callback()
    def _follower_image_callback(self, data):
        self.follower_image = data

    # def _follower_image_stat_callback(self, data):
    #     self.follower_image_stat = data

    def _conroller_twist(self, data):
        self.follower_controller_twist = data

    # TODO : rotation camera yaw and pitch info
    def _follower_camera_yaw_state(self, data):
        self.camera_yaw_state = data

    def _follower_camera_pitch_state(self, data):
        self.camera_pitch_state = data

    # def _lidar_callback(self, data):
    #     self.data_lidar = data

    def check_all_subscribers_ready(self):
        log.debug("START ALL SUBSCRIBERS READY")
        self._check_odom_ready()
        self._check_odom_target_ready()
        self._check_lidar_ready()
        self._check_target_status_ready()
        # TODO: self._check_image_ready()
        self._check_follower_image_ready()
        self._check_follower_controller_twist_ready()
        # TODO : rotation camera state
        self._check_follower_camera_yaw_state()
        self._check_follower_camera_pitch_state()
        # TODO : check lidar
        # self._check_lidar()



        log.debug("ALL SUBSCRIBERS READY")

    def _check_odom_ready(self):
        self.odom = None
        log.debug("Waiting for /default_robot/gazebo_ground_truth_odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/default_robot/gazebo_ground_truth_odom", Odometry, timeout=5.0)
                log.debug("Current /default_robot/gazebo_ground_truth_odom READY=>")
            except:
                log.error("Currnet /default_robot/gazebo_ground_truth_odom not ready yet, retrying for getting odom")

    def _check_odom_target_ready(self):
        self.odom_target = None
        log.debug("Waiting for /target_robot/target_ground_truth_odom to be READY...")
        while self.odom_target is None and not rospy.is_shutdown():
            try:
                self.odom_target = rospy.wait_for_message("/target_robot/target_ground_truth_odom", Odometry, timeout=5.0)
                log.debug("Current /target_robot/target_ground_truth_odom READY=>")
            except:
                log.error("Currnet /target_robot/target_ground_truth_odom not ready yet, retrying for getting odom target")

    def _check_lidar_ready(self):
        self.lidar = None
        log.debug("Waiting for /default_robot/velodyne_points2 to be READY...")
        while self.lidar is None and not rospy.is_shutdown():
            try:
                self.lidar = rospy.wait_for_message("/default_robot/velodyne_points2", PointCloud2, timeout=5.0)
                log.debug("Current /default_robot/velodyne_points2 READY=>")
            except:
                log.error("Current /default_robot/velodyne_points2 not ready yet, retrying for getting lidar")

    def _check_target_status_ready(self):
        self.target_status = None
        log.debug("Waiting for /target_robot/move_base/status to be READY...")
        while self.target_status is None and not rospy.is_shutdown():
            try:
                self.target_status = rospy.wait_for_message("/target_robot/move_base/status", GoalStatusArray, timeout=5.0)
                log.debug("Current /target_robot/move_base/status READY=>")
            except:
                log.error("Current /target_robot/move_base/status no ready yet, retrying for getting status")

    # TODO: _check_image_ready()

    def _check_follower_image_ready(self):
        self.follower_image = None
        log.debug("Waiting for /default_robot/rotating_camera/image_raw/compressed to be READY...")
        while self.follower_image is None and not rospy.is_shutdown():
            try:
                self.follower_image = rospy.wait_for_message("/default_robot/rotating_camera/image_raw/compressed", CompressedImage, timeout=5.0)
                log.debug("Current /default_robot/rotating_camera/image_raw/compressed READY=>")
            except:
                log.error("Current /default_robot/rotating_camera/image_raw/compressed no ready yet, retrying for getting status")

    def _check_follower_controller_twist_ready(self):
        self.follower_controller_twist = None
        log.debug("Waiting for /default_robot/mobile_base_controller/cmd_vel_out to be READY...")
        while self.follower_controller_twist is None and not rospy.is_shutdown():
            try:
                self.follower_controller_twist = rospy.wait_for_message("/default_robot/mobile_base_controller/cmd_vel_out", TwistStamped, timeout=5.0)
                log.debug("Current /default_robot/mobile_base_controller/cmd_vel_out READY=>")
            except:
                log.error("Current /default_robot/mobile_base_controller/cmd_vel_out no ready yet, retrying for getting status")

    def _check_follower_camera_yaw_state(self):
        self.camera_yaw_state = None
        log.debug("Waiting for /default_robot/camera_yaw_controller/state to be READY...")
        while self.camera_yaw_state is None and not rospy.is_shutdown():
            try:
                self.camera_yaw_state = rospy.wait_for_message("/default_robot/camera_yaw_controller/state", JointControllerState, timeout=5.0)
                log.debug("Current /default_robot/camera_yaw_controller/state READY=>")
            except:
                log.error("Current /default_robot/camera_yaw_controller/state no ready yet, retrying for getting status")

    def _check_follower_camera_pitch_state(self):
        self.camera_pitch_state = None
        log.debug("Waiting for /default_robot/camera_pitch_controller/state to be READY...")
        while self.camera_pitch_state is None and not rospy.is_shutdown():
            try:
                self.camera_pitch_state = rospy.wait_for_message("/default_robot/camera_pitch_controller/state", JointControllerState, timeout=5.0)
                log.debug("Current /default_robot/camera_pitch_controller/state READY=>")
            except:
                log.error("Current /default_robot/camera_pitch_controller/state no ready yet, retrying for getting status")

    # def _check_lidar(self):
    #     self.data_lidar = None
    #     log.debug("Waiting for /default_robot/velodyne_points2 to be READY...")
    #     while self.data_lidar is None and not rospy.is_shutdown():
    #         try:
    #             self.data_lidar = rospy.wait_for_message("/default_robot/velodyne_points2",
    #                                                              JointControllerState, timeout=5.0)
    #             log.debug("Current /default_robot/velodyne_points2 READY=>")
    #         except:
    #             log.error(
    #                 "Current /default_robot/velodyne_points2 no ready yet, retrying for getting status")
