import rospy
import matplotlib.pyplot as plt
import requests
import json

from pathlib import Path
from pyhocon import ConfigFactory

from src.arctic_gym import arctic_env_maker
from src.arctic_gym.arctic_env.arctic_env import ArcticEnv


class DebugEnv(ArcticEnv):

    def __init__(self, name, **config):
        super(DebugEnv, self).__init__(name, **config)

    def get_ssd_lead_information(self) -> dict:
        """
        Получение информации о распознанных объектах с камеры робота

        :return:
            словарь объектов с их границами на изображении
        """
        image = self.sub.get_from_follower_image()
        data = image.data

        results = requests.post(self.object_detection_endpoint, data=data)

        # ловим ошибки получения json
        try:
            results = json.loads(results.text)
        except json.decoder.JSONDecodeError:
            results = {}

        return results


if __name__ == '__main__':
    rospy.init_node("rl_client", anonymous=True)

    project_path = Path(__file__).resolve().parents[3]
    config_path = project_path.joinpath('config/config.conf')
    config = ConfigFactory.parse_file(config_path)

    env = arctic_env_maker(config.rl_agent.debug_config)

    objects = env.get_ssd_lead_information()
    env.get_lidar_points()

    length, other = env.calculate_length_to_leader(objects)

    print(length)