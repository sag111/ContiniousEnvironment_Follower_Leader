#!/usr/bin/env python
import rospy

from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Float64

from src.arctic_gym.utils.CustomFormatter import logger


log, formatter = logger(name='gazebo_pub', level='DEBUG')


class ArcticPublishers:
    """
    Publishers топиков ROS для арктики
    """
    def __init__(self):
        self.cmd_vel_pub = rospy.Publisher("/default_robot/mobile_base_controller/cmd_vel", Twist, queue_size=1)
        self.target_goal_pub = rospy.Publisher("/target_robot/move_base_simple/goal", PoseStamped, queue_size=1)
        self.teleport_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)

        self.cmd_vel_target_pub = rospy.Publisher("/target_robot/mobile_base_controller/cmd_vel_out", Twist, queue_size=1)

        self.camera_pitch_pub = rospy.Publisher('/default_robot/camera_pitch_controller/command', Float64,
                                                queue_size=1)
        self.camera_yaw_pub = rospy.Publisher('/default_robot/camera_yaw_controller/command', Float64,
                                                queue_size=1)

        # self.camera_rotate_manip_pub = rospy.Publisher('/default_robot/camera_yaw_controller/command', Float64,
        #                                       queue_size=1)
        # self.camera_rotate_manip_pub = rospy.Publisher('/default_robot/manip_stock_controller/command', Float64,
        #                                                 queue_size=1)


    # def set_camera_rotate_manip_pub(self, radian):
    #     rotate_value = Float64()
    #     rotate_value.data = radian
    #     # pitch_value.set_point = radian
    #     # TODO: управление камерой
    #     self._check_rotate_manip_pub_ready()
    #     self.camera_rotate_manip_pub.publish(rotate_value)
    #
    #     log.info(f'Change camera rotate: {radian}')

    def set_camera_pitch(self, radian):
        pitch_value = Float64()
        pitch_value.data = radian
        # pitch_value.set_point = radian
        # TODO: управление камерой
        self._check_camera_pitch_pub_ready()
        self.camera_pitch_pub.publish(pitch_value)

        log.info(f'Change camera pitch: {radian}')

    def set_camera_yaw(self, radian):
        yaw_value = Float64()
        yaw_value.data = radian

        # TODO: управление камерой
        self._check_camera_yaw_pub_ready()
        self.camera_yaw_pub.publish(yaw_value)

        log.info(f'Change camera yaw: {radian}')

    def move_base(self, linear_speed, angular_speed):
        """
        Управление ведомым роботом с помощью изменения линейной и угловой скорости
        default rostopic: /default_robot/mobile_base_controller/cmd_vel
        parmas: Twist()
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        log.debug(f"ArcticRobot Base Twist Cmd>>({linear_speed}, {angular_speed})")

        self._check_cmd_vel_pub_ready()
        self.cmd_vel_pub.publish(cmd_vel_value)

    def move_vel_target(self, linear_speed, angular_speed):
        """
        Управление ведомым роботом с помощью изменения линейной и угловой скорости
        default rostopic: /default_robot/mobile_base_controller/cmd_vel
        parmas: Twist()
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        log.debug(f"TargetRobot Base Twist Cmd>>({linear_speed}, {angular_speed})")

        self._check_cmd_vel_target_pub_ready()
        self.cmd_vel_target_pub.publish(cmd_vel_value)

    def move_target(self, x_position, y_position):
        """
        Управление ведущим роботом, движение в заданную точку на местности по координатам (x, y)
        default rostopic: /target_robot/move_base_simple/goal
        params: PoseStamped()
        """
        target_goal_value = PoseStamped()
        target_goal_value.header.frame_id = 'map'
        target_goal_value.pose.position.x = x_position
        target_goal_value.pose.position.y = y_position
        target_goal_value.pose.orientation.w = 1

        log.info(f"TargetRobot Base Goal Cmd>>({x_position}, {y_position})")

        self._check_target_goal_pub_ready()
        self.target_goal_pub.publish(target_goal_value)

    def teleport(self, model, point, quaternion):
        """
        Телепортация робота в точку, возможность перемещать как ведущего, так и ведомого роботов
        default rostopic: /gazebo/set_model_state
        params: ModelState()
        """
        point = Point(*point)
        quaternion = Quaternion(*quaternion)

        teleport_value = ModelState()
        teleport_value.model_name = model
        teleport_value.pose.position = point
        teleport_value.pose.orientation = quaternion

        log.debug(f"Teleport {model} into>> {point}")

        self._check_teleport_pub_ready()
        self.teleport_pub.publish(teleport_value)


    def _check_camera_pitch_pub_ready(self):
        """

        """
        rate = rospy.Rate(10)
        while self.camera_pitch_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            log.debug("camera_pitch_pub not ready yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

        log.debug("camera_pitch_pub connected")

    def _check_camera_yaw_pub_ready(self):
        """

        """
        rate = rospy.Rate(10)
        while self.camera_yaw_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            log.debug("camera_yaw_pub not ready yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

        log.debug("camera_yaw_pub connected")

    def _check_cmd_vel_pub_ready(self):
        """
        Проверка доступности publisher для self.move_base()
        """
        rate = rospy.Rate(10)
        while self.cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            log.debug("cmd_vel_pub not ready yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

        log.debug("_cmd_vel_pub connected")

    def _check_cmd_vel_target_pub_ready(self):
        """
        Проверка доступности publisher для self.move_base()
        """
        rate = rospy.Rate(10)
        while self.cmd_vel_target_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            log.debug("cmd_vel_target_pub not ready yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

        log.debug("_cmd_vel_target_pub connected")

    def _check_target_goal_pub_ready(self):
        """
        Проверка доступности publisher для self.move_target()
        """
        rate = rospy.Rate(10)
        while self.target_goal_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            log.debug("target_goal_pub not ready yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

        log.debug("target_goal_pub connected")

    def _check_teleport_pub_ready(self):
        """
        Проверка доступности publisher для self.teleport()
        """
        rate = rospy.Rate(10)
        while self.teleport_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            log.debug("teleport_pub not ready yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

        log.debug("teleport_pub connected")

