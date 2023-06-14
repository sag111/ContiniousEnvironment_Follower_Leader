import rospy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2
from control_msgs.msg import JointControllerState
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import TwistStamped

from pyhocon import ConfigTree


class Subscribers:

    def __init__(self, config: ConfigTree):
        # Координаты робота
        self.robot_odom_topic = config["topic.robot_odom"]
        rospy.Subscriber(self.robot_odom_topic, Odometry, self._odom_callback)
        # Координаты цели
        self.target_odom_topic = config["topic.target_odom"]
        rospy.Subscriber(self.target_odom_topic, Odometry, self._odom_target_callback)
        # Изображение с камеры робота
        self.robot_rotating_camera_topic = config["topic.robot_rotating_camera"]
        rospy.Subscriber(self.robot_rotating_camera_topic, CompressedImage, self._follower_image_callback)
        # Облако точек лидара робота
        self.robot_lidar_topic = config["topic.robot_lidar"]
        rospy.Subscriber(self.robot_lidar_topic, PointCloud2, self._lidar_callback)
        # Состояние угла рыскания камеры робота
        self.robot_camera_yaw_state_topic = config["topic.robot_camera_yaw_state"]
        rospy.Subscriber(self.robot_camera_yaw_state_topic, JointControllerState, self._follower_camera_yaw_callback)
        # Статус цели
        self.target_status_topic = config["topic.target_status"]
        rospy.Subscriber(self.target_status_topic, GoalStatusArray, self._target_status_callback)
        # Статус робота
        self.robot_status_move_to_topic = config["topic.robot_status_move_to"]
        rospy.Subscriber(self.robot_status_move_to_topic, GoalStatusArray, self._move_to_status_callback)

        self.check_all_subscribers_ready()

    def _odom_callback(self, data):
        self.odom = data

    def get_odom(self):
        """
        Получение координат робота
        """
        return self.odom

    def _odom_target_callback(self, data):
        self.odom_target = data

    def get_odom_target(self):
        """
        Получение координат цели
        """
        return self.odom_target

    def _follower_image_callback(self, data):
        self.follower_image = data

    def get_from_follower_image(self):
        """
        Получение изображения с камеры робота
        """
        return self.follower_image

    def _lidar_callback(self, data):
        self.lidar = data

    def get_lidar(self):
        """
        Получение облака точек лидара робота
        """
        return self.lidar

    def _follower_camera_yaw_callback(self, data):
        self.camera_yaw_state = data

    def get_camera_yaw_state(self):
        """
        Получение состояния угла рыскания камеры робота
        """
        return self.camera_yaw_state

    def _target_status_callback(self, data):
        self.target_status = data

    def get_target_status(self):
        """
        Получение статуса цели
        """
        return self.target_status

    def _move_to_status_callback(self, data):
        self.move_to_status = data

    def get_move_to_status(self):
        """
        Получение статуса робота
        """
        return self.move_to_status

    def check_all_subscribers_ready(self):
        """
        Проверка подписок
        """
        self._check_odom_ready()
        self._check_odom_target_ready()
        self._check_follower_image_ready()
        self._check_lidar_ready()
        self._check_follower_camera_yaw_ready()
        self._check_target_status_ready()

    def _check_odom_ready(self):
        """
        Проверка получения координат робота
        """
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message(self.robot_odom_topic, Odometry)
            except:
                rospy.logerr(f"Current {self.robot_odom_topic} not ready yet, retrying for getting odom")

    def _check_odom_target_ready(self):
        """
        Проверка получения координат цели
        """
        self.odom_target = None
        while self.odom_target is None and not rospy.is_shutdown():
            try:
                self.odom_target = rospy.wait_for_message(self.target_odom_topic, Odometry)
            except:
                rospy.logerr(f"Current {self.target_odom_topic} not ready yet, retrying for getting odom target")

    def _check_follower_image_ready(self):
        """
        Проверка получения изображения с камеры робота
        """
        self.follower_image = None
        while self.follower_image is None and not rospy.is_shutdown():
            try:
                self.follower_image = rospy.wait_for_message(self.robot_rotating_camera_topic, CompressedImage)
            except:
                rospy.logerr(f"Current {self.robot_rotating_camera_topic} no ready yet, retrying for getting status")

    def _check_lidar_ready(self):
        """
        Проверка получения облака точек лидара робота
        """
        self.lidar = None
        while self.lidar is None and not rospy.is_shutdown():
            try:
                self.lidar = rospy.wait_for_message(self.robot_lidar_topic, PointCloud2)
            except:
                rospy.logerr(f"Current {self.robot_lidar_topic} not ready yet, retrying for getting lidar")

    def _check_follower_camera_yaw_ready(self):
        """
        Проверка получения состояния угла рыскания камеры робота
        """
        self.camera_yaw_state = None
        while self.camera_yaw_state is None and not rospy.is_shutdown():
            try:
                self.camera_yaw_state = rospy.wait_for_message(self.robot_camera_yaw_state_topic, JointControllerState)
            except:
                rospy.logerr(f"Current {self.robot_camera_yaw_state_topic} no ready yet, retrying for getting status")

    def _check_target_status_ready(self):
        """
        Проверка получения статуса цели
        """
        self.target_status = None
        while self.target_status is None and not rospy.is_shutdown():
            try:
                self.target_status = rospy.wait_for_message(self.target_status_topic, GoalStatusArray)
            except:
                rospy.logerr(f"Current {self.target_status_topic} no ready yet, retrying for getting status")

    def _check_get_move_to_status_ready(self):
        """
        Проверка получения статуса робота
        """
        self.move_to_status = None
        while self.move_to_status is None and not rospy.is_shutdown():
            try:
                self.move_to_status = rospy.wait_for_message(self.robot_status_move_to_topic, TwistStamped)
            except:
                rospy.logerr(f"Current {self.robot_status_move_to_topic} no ready yet, retrying for getting status")


if __name__ == '__main__':
    from pathlib import Path
    from pyhocon import ConfigFactory

    project_path = Path(__file__).resolve().parents[3]
    config_path = project_path.joinpath('config/config.conf')
    config = ConfigFactory.parse_file(config_path)

    rospy.init_node('test_subscribers', anonymous=True)
    sub = Subscribers(config)
