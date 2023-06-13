#!/usr/bin/env python
import rospy

from std_srvs.srv import Empty
from src.arctic_gym.utils.CustomFormatter import logger


log, formatter = logger(name='gazebo_srv', level='INFO')


class GazeboConnection:
    def __init__(self):
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

    def reset_world(self):
        """
        Proxy service для сброса среды Gazebo к начальному состоянию на момент запуска
        """

        log.info(formatter.colored_logs("Reset Gazebo world", "yellow"))
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            log.error("/gazebo/reset_world service call failed")
