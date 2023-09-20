import rospy
import numpy as np

from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from actionlib_msgs.msg import GoalID
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker

from pyhocon import ConfigTree
from typing import Optional, Any
from math import sin, cos


class Publishers:

    def __init__(self, config: ConfigTree, rate: int = 10):
        self.rate = rospy.Rate(rate)
        # Camera pitch angle control
        self.camera_pitch_pub = rospy.Publisher(config["topic.robot_camera_pitch"], Float64, queue_size=1)
        # Camera yaw angle control
        self.camera_yaw_pub = rospy.Publisher(config["topic.robot_camera_yaw"], Float64, queue_size=1)
        # Teleport the model
        self.teleport_pub = rospy.Publisher(config["topic.teleport"], ModelState, queue_size=1)
        # The leader control by coordinates based on path planning
        self.target_goal_pub = rospy.Publisher(config["topic.target_goal"], PoseStamped, queue_size=1)
        # The agent velocity control
        self.cmd_vel_pub = rospy.Publisher(config["topic.robot_cmd_vel"], Twist, queue_size=1)
        # Cancel the leader movement
        self.target_cancel_action_pub = rospy.Publisher(config["topic.target_cancel"], GoalID, queue_size=1)
        # The agent control by coordinates based on path planning
        self.default_goal_pub = rospy.Publisher(config["topic.robot_goal"], PoseStamped, queue_size=1)

        # External topic, publishes the agent path
        self.follower_path_pub = rospy.Publisher(config["topic.robot_path"], Path, queue_size=1)
        self.follower_path = Path()
        # External topic, publishes the leader path
        self.target_path_pub = rospy.Publisher(config["topic.target_path"], Path, queue_size=1)
        self.target_path = Path()

        # External topic for safe zone boundaries
        self.corridor_marker_pub = rospy.Publisher("/external/corridor", MarkerArray, queue_size=1)

    def set_camera_pitch(self, radian: float):
        """
        Camera pitch angle control

        :param radian: angle in radians
        """
        pitch_value = Float64()
        pitch_value.data = radian

        self._check_camera_pitch_pub_ready()
        self.camera_pitch_pub.publish(pitch_value)

    def _check_camera_pitch_pub_ready(self):
        """
        Checking the connection to the camera pitch angle topic
        """

        while self.camera_pitch_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def set_camera_yaw(self, radian: float):
        """
        Camera yaw angle control

        :param radian: angle in radians
        """
        yaw_value = Float64()
        yaw_value.data = radian

        self._check_camera_yaw_pub_ready()
        self.camera_yaw_pub.publish(yaw_value)

    def _check_camera_yaw_pub_ready(self):
        """
        Checking the connection to the camera yaw angle topic
        """
        while self.camera_yaw_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def teleport(self, model: str, point: Any, quaternion: Any):
        """
        Teleport the model

        :param model: model name
        :param point: coordinates to move
        :param quaternion: angle of rotation
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
        Checking the connection to the model state topic
        """
        while self.teleport_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def move_target(self, x_position: float, y_position: float, phi: int = 0):
        """
        The leader control by coordinates based on path planning

        :param x_position: absolute X coordinate
        :param y_position: absolute Y coordinate
        :param phi: rotation angle in degrees
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
        Checking the connection to the topic of the final coordinates of the leader
        """
        while self.target_goal_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def move_base(self, linear_speed: float, angular_speed: float):
        """
        the agent speed control

        :param linear_speed: linear velocity
        :param angular_speed: angular velocity
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        self._check_cmd_vel_pub_ready()
        self.cmd_vel_pub.publish(cmd_vel_value)

    def _check_cmd_vel_pub_ready(self):
        """
        Checking the connection to the agent velocity topic
        """
        while self.cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def target_cancel_action(self):
        """
        Cancel the leader movement
        """
        target_cancel_action_value = GoalID()

        self._check_target_cancel_action_pub_ready()
        self.target_cancel_action_pub.publish(target_cancel_action_value)

    def _check_target_cancel_action_pub_ready(self):
        """
        Checking the connection with the leader movement cancellation topic
        """
        while self.target_cancel_action_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def move_default(self, x_position: float, y_position: float, phi: int = 0):
        """
        The agent control by coordinates based on path planning

        :param x_position: absolute X coordinate
        :param y_position: absolute Y coordinate
        :param phi: rotation angle in degrees
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
       Checking the connection to the topic of the final coordinates of the agent
        """
        while self.default_goal_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def update_follower_path(self, x: Optional[float] = None, y: Optional[float] = None):
        """
        the agent path update

        :param x: absolute X coordinate of the path
        :param y: absolute Y coordinate of the path
        """
        self.follower_path.header.frame_id = "map"

        pose_stamp = PoseStamped()
        pose_stamp.header.frame_id = "map"
        pose_stamp.header.stamp = rospy.Time.now()

        if x and y:
            pose_stamp.pose.position.x = x
            pose_stamp.pose.position.y = y
            self.follower_path.poses.append(pose_stamp)
        else:
            self.follower_path.poses.append(pose_stamp)
            self.follower_path.poses.clear()

        self._check_follower_path_pub_ready()
        self.follower_path_pub.publish(self.follower_path)

    def _check_follower_path_pub_ready(self):
        """
        Checking the connection to the topic for publishing the agent path
        """
        while self.follower_path_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def update_target_path(self, x: Optional[float] = None, y: Optional[float] = None):
        """
        the target path update

        :param x: absolute X coordinate of the path
        :param y: absolute Y coordinate of the path
        """
        self.target_path.header.frame_id = "map"

        pose_stamp = PoseStamped()
        pose_stamp.header.frame_id = "map"
        pose_stamp.header.stamp = rospy.Time.now()

        if x and y:
            pose_stamp.pose.position.x = x
            pose_stamp.pose.position.y = y
            self.target_path.poses.append(pose_stamp)
        else:
            self.target_path.poses.clear()

        self._check_target_path_pub_ready()
        self.target_path_pub.publish(self.target_path)

    def _check_target_path_pub_ready(self):
        """
        Checking the connection to the topic for publishing the leader path
        """
        while self.target_path_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def update_corridor(self, corridor: list):
        """
        safe zone points update

        :param corridor: list of safe zone points sequences
        """
        markers = []
        idx = 0
        for pair in corridor:
            for point in pair:
                mark = Marker()

                mark.header.frame_id = "map"
                mark.id = idx
                mark.type = 1

                mark.pose.position.x = point[0]
                mark.pose.position.y = point[1]

                mark.pose.orientation.w = 1.0

                mark.scale.x = 0.1
                mark.scale.y = 0.1
                mark.scale.z = 0.1

                mark.color.r = 0.5
                mark.color.a = 1

                markers.append(mark)

                idx += 1

        mark_array = MarkerArray()
        mark_array.markers = markers

        self._check_corridor_marker_pub_ready()
        self.corridor_marker_pub.publish(mark_array)

    def _check_corridor_marker_pub_ready(self):
        """
        Checking the connection to the safe zone topic
        """
        while self.corridor_marker_pub.get_num_connections() == 0 and not rospy.is_shutdown():
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
