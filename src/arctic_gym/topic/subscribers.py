import rospy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2
from control_msgs.msg import JointControllerState
from actionlib_msgs.msg import GoalStatusArray
from nav_msgs.msg import Path
from gazebo_msgs.msg import ModelStates

from pyhocon import ConfigTree


class Subscribers:

    def __init__(self, config: ConfigTree):
        # the agent coordinates
        self.robot_odom_topic = config["topic.robot_odom"]
        rospy.Subscriber(self.robot_odom_topic, Odometry, self._odom_callback)
        # the leader coordinates
        self.target_odom_topic = config["topic.target_odom"]
        rospy.Subscriber(self.target_odom_topic, Odometry, self._odom_target_callback)
        # the agent camera image
        self.robot_rotating_camera_topic = config["topic.robot_rotating_camera"]
        rospy.Subscriber(self.robot_rotating_camera_topic, CompressedImage, self._follower_image_callback)
        # the agent point cloud from lidar
        self.robot_lidar_topic = config["topic.robot_lidar"]
        rospy.Subscriber(self.robot_lidar_topic, PointCloud2, self._lidar_callback)
        # the agent camera yaw angle status
        self.robot_camera_yaw_state_topic = config["topic.robot_camera_yaw_state"]
        rospy.Subscriber(self.robot_camera_yaw_state_topic, JointControllerState, self._follower_camera_yaw_callback)
        # the leader status
        self.target_status_topic = config["topic.target_status"]
        rospy.Subscriber(self.target_status_topic, GoalStatusArray, self._target_status_callback)
        # the agent status
        self.robot_status_move_to_topic = config["topic.robot_status_move_to"]
        rospy.Subscriber(self.robot_status_move_to_topic, GoalStatusArray, self._move_to_status_callback)
        # the leader path
        self.target_path_topic = config["topic.target_path"]
        rospy.Subscriber(self.target_path_topic, Path, self._target_path_callback)
        # the agent path
        self.robot_path_topic = config["topic.robot_path"]
        rospy.Subscriber(self.robot_path_topic, Path, self._robot_path_callback)
        # gazebo model states
        self.gazebo_states_topic = config["topic.model_states"]
        rospy.Subscriber(self.gazebo_states_topic, ModelStates, self._gazebo_states_callback)

        self.check_all_subscribers_ready()

    def _odom_callback(self, data):
        self.odom = data

    def get_odom(self):
        """
        gets the agent odometry
        """
        return self.odom

    def _odom_target_callback(self, data):
        self.odom_target = data

    def get_odom_target(self):
        """
        gets the leader odometry
        """
        return self.odom_target

    def _follower_image_callback(self, data):
        self.follower_image = data

    def get_from_follower_image(self):
        """
        gets image from agent's camera
        """
        return self.follower_image

    def _lidar_callback(self, data):
        self.lidar = data

    def get_lidar(self):
        """
        gets point cloud from agent's lidar
        """
        return self.lidar

    def _follower_camera_yaw_callback(self, data):
        self.camera_yaw_state = data

    def get_camera_yaw_state(self):
        """
        gets the yaw angle state of the agent's camera
        """
        return self.camera_yaw_state

    def _target_status_callback(self, data):
        self.target_status = data

    def get_target_status(self):
        """
        gets the leader status
        """
        return self.target_status

    def _move_to_status_callback(self, data):
        self.move_to_status = data

    def get_move_to_status(self):
        """
        gets the agent status
        """
        return self.move_to_status

    def _target_path_callback(self, data):
        self.target_path = data

    def get_target_path(self):
        """
        gets the leader path
        """
        return self.target_path

    def _robot_path_callback(self, data):
        self.robot_path = data

    def get_robot_path(self):
        """
        gets the agent path
        """
        return self.robot_path

    def _gazebo_states_callback(self, data):
        self.model_states = data

    def get_model_states(self):
        """
        gets gazebo model states
        """
        return self.model_states

    def check_all_subscribers_ready(self):
        """
        checks subscribers
        """
        self._check_odom_ready()
        self._check_odom_target_ready()
        self._check_follower_image_ready()
        self._check_lidar_ready()
        self._check_follower_camera_yaw_ready()
        self._check_target_status_ready()
        self._check_gazebo_states_ready()

    def _check_odom_ready(self):
        """
        Checking receipt of agent odometry
        """
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message(self.robot_odom_topic, Odometry)
            except:
                rospy.logerr(f"Current {self.robot_odom_topic} not ready yet, retrying for getting odom")

    def _check_odom_target_ready(self):
        """
        Checking receipt of leader odometry
        """
        self.odom_target = None
        while self.odom_target is None and not rospy.is_shutdown():
            try:
                self.odom_target = rospy.wait_for_message(self.target_odom_topic, Odometry)
            except:
                rospy.logerr(f"Current {self.target_odom_topic} not ready yet, retrying for getting odom target")

    def _check_follower_image_ready(self):
        """
        Checking the image received from the agent's camera
        """
        self.follower_image = None
        while self.follower_image is None and not rospy.is_shutdown():
            try:
                self.follower_image = rospy.wait_for_message(self.robot_rotating_camera_topic, CompressedImage)
            except:
                rospy.logerr(f"Current {self.robot_rotating_camera_topic} no ready yet, retrying for getting status")

    def _check_lidar_ready(self):
        """
        Checking the receipt of the agent's lidar point cloud
        """
        self.lidar = None
        while self.lidar is None and not rospy.is_shutdown():
            try:
                self.lidar = rospy.wait_for_message(self.robot_lidar_topic, PointCloud2)
            except:
                rospy.logerr(f"Current {self.robot_lidar_topic} not ready yet, retrying for getting lidar")

    def _check_follower_camera_yaw_ready(self):
        """
        Checking the receipt of the yaw angle state of the agent camera
        """
        self.camera_yaw_state = None
        while self.camera_yaw_state is None and not rospy.is_shutdown():
            try:
                self.camera_yaw_state = rospy.wait_for_message(self.robot_camera_yaw_state_topic, JointControllerState)
            except:
                rospy.logerr(f"Current {self.robot_camera_yaw_state_topic} no ready yet, retrying for getting status")

    def _check_target_status_ready(self):
        """
        Checking the leader status receipt
        """
        self.target_status = None
        while self.target_status is None and not rospy.is_shutdown():
            try:
                self.target_status = rospy.wait_for_message(self.target_status_topic, GoalStatusArray)
            except:
                rospy.logerr(f"Current {self.target_status_topic} no ready yet, retrying for getting status")

    def _check_get_move_to_status_ready(self):
        """
        Checking the agent status receipt
        """
        self.move_to_status = None
        while self.move_to_status is None and not rospy.is_shutdown():
            try:
                self.move_to_status = rospy.wait_for_message(self.robot_status_move_to_topic, GoalStatusArray)
            except:
                rospy.logerr(f"Current {self.robot_status_move_to_topic} no ready yet, retrying for getting status")

    def _check_gazebo_states_ready(self):
        """
        Checking the gazebo model states receipt
        """
        self.model_states = None
        while self.model_states is None and not rospy.is_shutdown():
            try:
                self.model_states = rospy.wait_for_message(self.gazebo_states_topic, ModelStates)
            except:
                rospy.logerr(f"Current {self.gazebo_states_topic} no ready yet, retrying for getting status")


if __name__ == '__main__':
    from pathlib import Path
    from pyhocon import ConfigFactory

    project_path = Path(__file__).resolve().parents[3]
    config_path = project_path.joinpath('config/config.conf')
    config = ConfigFactory.parse_file(config_path)

    rospy.init_node('test_subscribers', anonymous=True)
    sub = Subscribers(config)
