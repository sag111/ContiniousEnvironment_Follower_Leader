import rospy
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import cv2

from pathlib import Path
from pyhocon import ConfigFactory
from math import atan, tan, sqrt, cos, sin

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

    @staticmethod
    def get_crossed(objects: list) -> list:
        """
        Выделение машины из объектов и пересекающихся с ней объектов

        :param objects: список объектов полученных из object detection

        :return:
            Список объектов пересекающихся с машиной
        """
        cars = [x for x in objects if x['name'] == "car"]

        # если определилось несколько машин, находим машину с наибольшей площадью bounding box
        max_square = 0
        car = {}
        for one in cars:
            # удаляем машину из общего списка объектов
            objects.remove(one)
            square = (one['xmax'] - one['xmin']) * (one['ymax'] - one['ymin'])

            if square > max_square:
                car = one
                max_square = square

        crossed_objects = [car, ]
        for obj in objects:
            x1_c = car['xmin']
            x2_c = car['xmax']
            y1_c = car['ymin']
            y2_c = car['ymax']

            x1_o = obj['xmin']
            x2_o = obj['xmax']
            y1_o = obj['ymin']
            y2_o = obj['ymax']

            # находим пересечение bounding box объектов с bounding box машины:
            if (x1_c < x2_o and x2_c > x1_o) and (y1_c < y2_o and y2_c > y1_o):
                crossed_objects.append(obj)

        return crossed_objects

    @staticmethod
    def calculate_points_angles_objects(objects: list) -> list:
        """
        функция вычисления углов по значениям bounding box

        :param objects: список объектов

        :return:
            Список объектов с углам для ориентации
            [p_theta1, p_phi1, p_theta2, p_phi2]
        """
        angles = []
        for obj in objects:
            xmin = obj['xmin']
            ymin = obj['ymin']
            xmax = obj['xmax']
            ymax = obj['ymax']

            xmin = xmin - 5
            xmax = xmax + 5

            p_theta1 = atan((2 * xmin - 640) / 640 * tan(80 / 2))
            p_phi1 = atan(-((2 * ymin - 480) / 480) * tan(64 / 2))
            p_theta2 = atan((2 * xmax - 640) / 640 * tan(80 / 2))
            p_phi2 = atan(-((2 * ymax - 480) / 480) * tan(64 / 2))

            angles.append({
                'name': obj['name'],
                'theta1': p_theta1,   # 0
                'phi1': p_phi1,       # 1
                'theta2': p_theta2,   # 2
                'phi2': p_phi2        # 3
            })

        return angles

    def calculate_lidar_points(self, objects):

        maxd = 30
        # угол постоянного поворота камеры по горизонталиget_crossed
        camera_yaw = self.sub.get_camera_yaw_state().process_value

        insiders = []
        for obj in objects:

            self.get_lidar_points()
            points_inside = []
            for i in self.lidar_points:
                dist = np.linalg.norm(i[:2])

                k1_x = (tan(np.deg2rad(-40)) + tan(camera_yaw)) * i[0]
                k2_x = (tan(np.deg2rad(40)) + tan(camera_yaw)) * i[0]

                theta2_x = (tan(obj["theta2"]) + tan(camera_yaw)) * i[0]
                theta1_x = (tan(obj["theta1"]) + tan(camera_yaw)) * i[0]

                phi2_x = tan(obj["phi2"]) * i[0]
                phi1_x = tan(obj["phi1"]) * i[0]

                if dist <= maxd and k1_x <= i[1] <= k2_x and theta2_x <= i[1] <= theta1_x and phi2_x <= i[2] <= phi1_x:
                    points_inside.append(i[:3])

            points_inside = np.array(points_inside)

            # ax = plt.axes(projection="3d")
            # ax.scatter(points_inside[:, 0], points_inside[:, 1], points_inside[:, 2])
            # ax.scatter(0, 0, 0, 'x')
            # plt.show()

            norms = np.linalg.norm(points_inside, axis=1)
            insiders.append({
                "name": obj["name"],
                "data": norms
            })

        diff = np.setdiff1d(insiders[0]["data"], insiders[2]["data"])

        diff = np.setdiff1d(diff, insiders[2]["data"])

        plt.hist(diff)
        plt.show()

        for i in insiders:
            plt.hist(i["data"], label=i["name"], histtype='step')

        plt.legend()
        plt.show()


if __name__ == '__main__':
    rospy.init_node("rl_client", anonymous=True)

    project_path = Path(__file__).resolve().parents[3]
    config_path = project_path.joinpath('config/config.conf')
    config = ConfigFactory.parse_file(config_path)

    env = arctic_env_maker(config.rl_agent.debug_config)

    objs = env.get_ssd_lead_information()
    cross = env.get_crossed(objs)
    angles = env.calculate_points_angles_objects(cross)

    env.calculate_lidar_points(angles)

    # car = [x for x in objects if x["name"] == "car"][0]
    #
    # objects.remove(car)
    #
    # crossed_objects = [car, ]
    #
    # for obj in objects:
    #     if (car['xmin'] < obj['xmax'] and car['xmax'] > obj['xmin']) and (car['ymin'] < obj['ymax'] and car['ymax'] > obj['ymin']):
    #         crossed_objects.append(obj)
    #
    # print(env.calculate_points_angles_objects(crossed_objects))
