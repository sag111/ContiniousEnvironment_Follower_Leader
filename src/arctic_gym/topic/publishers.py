import rospy
import numpy as np

from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from actionlib_msgs.msg import GoalID

from pyhocon import ConfigTree
from typing import Optional
from math import sin, cos


class Publishers:

    def __init__(self, config: ConfigTree, rate: int = 10):
        self.rate = rospy.Rate(rate)
        # Управление углом наклона камеры
        self.camera_pitch_pub = rospy.Publisher(config["topic.robot_camera_pitch"], Float64, queue_size=1)
        # Управление углом рыскания камеры
        self.camera_yaw_pub = rospy.Publisher(config["topic.robot_camera_yaw"], Float64, queue_size=1)
        # Перемещение модели
        self.teleport_pub = rospy.Publisher(config["topic.teleport"], ModelState, queue_size=1)
        # Управление целью по координатам
        self.target_goal_pub = rospy.Publisher(config["topic.target_goal"], PoseStamped, queue_size=1)
        # Управление скоростью робота
        self.cmd_vel_pub = rospy.Publisher(config["topic.robot_cmd_vel"], Twist, queue_size=1)
        # Отмена движения цели
        self.target_cancel_action_pub = rospy.Publisher(config["topic.target_cancel"], GoalID, queue_size=1)
        # Управление роботом по координатам
        self.default_goal_pub = rospy.Publisher(config["topic.robot_goal"], PoseStamped, queue_size=1)

    def set_camera_pitch(self, radian: float):
        """
        Управление углом наклона камеры

        :param radian: угол в радианах
        """
        pitch_value = Float64()
        pitch_value.data = radian

        self._check_camera_pitch_pub_ready()
        self.camera_pitch_pub.publish(pitch_value)

    def _check_camera_pitch_pub_ready(self):
        """
        Проверка соединения с топиком угла наклона камеры
        """

        while self.camera_pitch_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def set_camera_yaw(self, radian: float):
        """
        Управление углом рыскания камеры

        :param radian: угол в радианах
        """
        yaw_value = Float64()
        yaw_value.data = radian

        self._check_camera_yaw_pub_ready()
        self.camera_yaw_pub.publish(yaw_value)

    def _check_camera_yaw_pub_ready(self):
        """
        Проверка соединения с топиком угла рыскания камеры
        """
        while self.camera_yaw_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def teleport(self, model: str, point: Optional[list, Point], quaternion: Optional[list, Quaternion]):
        """
        Перемещение модели

        :param model: название модели
        :param point: координаты для перемещения
        :param quaternion: угол поворота
        """
        try:
            point_msg = Point(*point)
        except TypeError:
            point_msg = point

        try:
            quaternion_msg = Quaternion(*quaternion)
        except TypeError:
            quaternion_msg = quaternion

        teleport_value = ModelState()
        teleport_value.model_name = model
        teleport_value.pose.position = point_msg
        teleport_value.pose.orientation = quaternion_msg

        self._check_teleport_pub_ready()
        self.teleport_pub.publish(teleport_value)

        rospy.sleep(0.1)

    def _check_teleport_pub_ready(self):
        """
        Проверка соединения с топиком состояния модели
        """
        while self.teleport_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def move_target(self, x_position: float, y_position: float, phi: int = 0):
        """
        Управление целью

        :param x_position: абсолютная координата X
        :param y_position: абсолютная координата Y
        :param phi: угол поворота в градусах
        """
        target_goal_value = PoseStamped()
        target_goal_value.header.frame_id = 'map'
        target_goal_value.pose.position.x = x_position
        target_goal_value.pose.position.y = y_position

        r = np.deg2rad(phi/2)
        target_goal_value.pose.orientation.z = sin(r)
        target_goal_value.pose.orientation.w = cos(r)

        self._check_target_goal_pub_ready()
        self.target_goal_pub.publish(target_goal_value)

    def _check_target_goal_pub_ready(self):
        """
        Проверка соединения с топиком конечных координат цели
        """
        while self.target_goal_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def move_base(self, linear_speed: float, angular_speed: float):
        """
        Управление скоростью робота

        :param linear_speed: линейная скорость
        :param angular_speed: угловая скорость
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        self._check_cmd_vel_pub_ready()
        self.cmd_vel_pub.publish(cmd_vel_value)

    def _check_cmd_vel_pub_ready(self):
        """
        Проверка соединения с топиком скорости робота
        """
        while self.cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def target_cancel_action(self):
        """
        Отмена движения цели
        """
        target_cancel_action_value = GoalID()

        self._check_target_cancel_action_pub_ready()
        self.target_cancel_action_pub.publish(target_cancel_action_value)

    def _check_target_cancel_action_pub_ready(self):
        """
        Проверка соединения с топиком отмены движения цели
        """
        while self.target_cancel_action_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def move_default(self, x_position: float, y_position: float, phi: int = 0):
        """
        Управление роботом по координатам

        :param x_position: абсолютная координата X
        :param y_position: абсолютная координата Y
        :param phi: угол поворота в градусах
        """
        default_goal_value = PoseStamped()
        default_goal_value.header.frame_id = 'map'
        default_goal_value.pose.position.x = x_position
        default_goal_value.pose.position.y = y_position

        r = np.deg2rad(phi/2)
        default_goal_value.pose.orientation.z = sin(r)
        default_goal_value.pose.orientation.w = cos(r)

        self._check_default_goal_pub_ready()
        self.default_goal_pub.publish(default_goal_value)

    def _check_default_goal_pub_ready(self):
        """
        Проверка соединения с топиком для движения робота по координатам
        """
        while self.default_goal_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass


if __name__ == '__main__':
    from pathlib import Path
    from pyhocon import ConfigFactory

    project_path = Path(__file__).resolve().parents[3]
    config_path = project_path.joinpath('config/config.conf')
    config = ConfigFactory.parse_file(config_path)

    rospy.init_node('test_publishers', anonymous=True)
    pub = Publishers(config)
