import rospy
import numpy as np

from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

from pyhocon import ConfigTree
from math import sin, cos


class Publishers:

    def __init__(self, config: ConfigTree, rate: float = 10.0):
        self.rate = rospy.Rate(rate)
        # Управление углом наклона камеры
        self.camera_pitch_pub = rospy.Publisher(config["topic.robot_camera_pitch"], Float64, queue_size=1)
        # Управление углом рыскания камеры
        self.camera_yaw_pub = rospy.Publisher(config["topic.robot_camera_yaw"], Float64, queue_size=1)
        # Перемещение модели
        self.teleport_pub = rospy.Publisher(config["topic.teleport"], ModelState, queue_size=1)
        # Управление целью
        self.target_goal_pub = rospy.Publisher(config["topic.target_goal"], PoseStamped, queue_size=1)
        # Управление скоростью робота
        self.cmd_vel_pub = rospy.Publisher(config["topic.robot_cmd_vel"], Twist, queue_size=1)

    def set_camera_pitch(self, radian):
        """
        Управление углом наклона камеры
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

    def set_camera_yaw(self, radian):
        """
        Управление углом рыскания камеры
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

    def teleport(self, model, point, quaternion):
        """
        Перемещение модели
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

    def _check_teleport_pub_ready(self):
        """
        Проверка соединения с топиком состояния модели
        """
        while self.teleport_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def move_target(self, x_position, y_position, phi=0):
        """
        Управление целью
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

    def move_base(self, linear_speed, angular_speed):
        """
        Управление скоростью робота
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


if __name__ == '__main__':
    from pathlib import Path
    from pyhocon import ConfigFactory

    project_path = Path(__file__).resolve().parents[3]
    config_path = project_path.joinpath('config/config.conf')
    config = ConfigFactory.parse_file(config_path)

    rospy.init_node('test_publishers', anonymous=True)
    pub = Publishers(config)
