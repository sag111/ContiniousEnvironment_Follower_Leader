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

    def get_ssd_lead_information(self) -> list:
        """
        Получение информации о распознанных объектах с камеры робота

        :return:
            список объектов с их границами на изображении
        """
        image = self.sub.get_from_follower_image()
        data = image.data

        results = requests.post(self.object_detection_endpoint, data=data)

        # ловим ошибки получения json
        try:
            results = json.loads(results.text)
        except json.decoder.JSONDecodeError:
            results = []

        return results

    @staticmethod
    def calculate_points_angles_objects(obj: dict,
                                        width: int = 640,
                                        height: int = 480,
                                        hov: float = 80.0,
                                        fov: float = 64.0) -> dict:
        """
        Вычисление углов по значениям bounding box, вычисления основано на цилиндрической системе координат,
        центр изображения - центр системы координат

        :param obj: словарь объекта с ключами name - имя объекта; xmin, xmax, ymin, ymax - координаты bounding box
        :param width: ширина изображения в пикселях
        :param height: высота изображения в пикселях
        :param hov: горизонтальный угол обзора камеры
        :param fov: вертикальный угол обзора камеры

        :return:
            Словарь с граничными углами области объекта по вертикали (phi1, phi2) и горизонтали (theta1, theta2)
        """

        xmin = obj['xmin']
        xmax = obj['xmax']
        ymin = obj['ymin']
        ymax = obj['ymax']

        theta1 = atan((2 * xmin - width) / width * tan(hov / 2))
        theta2 = atan((2 * xmax - width) / width * tan(hov / 2))

        phi1 = atan(-((2 * ymin - height) / height) * tan(fov / 2))
        phi2 = atan(-((2 * ymax - height) / height) * tan(fov / 2))

        return {
            "theta1": theta1,
            "theta2": theta2,
            "phi1": phi1,
            "phi2": phi2
        }

    def calculate_lidar_points_inside(self, angles: dict, lidar: list, maxd: int = 50) -> dict:
        """
        Вычисление точек лидара находящихся внутри и снаружи области ограниченной углами angles

        :param angles: словарь с углами theta1, theta2, phi1, phi2
        :param lidar: облако точек лидара
        :param maxd: максимальное "видимое" расстояние для точек лидара

        :return:
            Словарь с координатами точек лидара: in - точки внутри области, out - точки снаружи области
        """
        camera_yaw = self.sub.get_camera_yaw_state().process_value

        points_inside = []
        points_outside = []
        for i in lidar:
            dist = np.linalg.norm(i[:2])

            k1_x = (tan(np.deg2rad(-40)) + tan(camera_yaw)) * i[0]
            k2_x = (tan(np.deg2rad(40)) + tan(camera_yaw)) * i[0]

            theta2_x = (tan(angles["theta2"]) + tan(camera_yaw)) * i[0]
            theta1_x = (tan(angles["theta1"]) + tan(camera_yaw)) * i[0]

            phi2_x = tan(angles["phi2"]) * i[0]
            phi1_x = tan(angles["phi1"]) * i[0]

            if dist <= maxd and k1_x <= i[1] <= k2_x and theta2_x <= i[1] <= theta1_x and phi2_x <= i[2] <= phi1_x:
                points_inside.append(list(i))
            elif 2 <= dist <= maxd:
                points_outside.append(list(i))

        return {
            "in": points_inside,
            "out": points_outside
        }

    def calculate_car_distance_v1(self, detected: list):
        """
        Определение расстояния до машины. Выбираем bounding box машины и пересекающиеся с ней объекты, вырезаем точки
        лидара, которые располагаются в области пересекающихся объектов, по оставшимся точкам вычисляем расстояние

        :param detected: список распознанных объектов, каждый объект представляет собой словарь со следующими ключами
            name - имя объекта; xmin, xmax, ymin, ymax - координаты bounding box
        """

        cars = [x for x in detected if x['name'] == "car"]

        # если определилось несколько машин, находим машину с наибольшей площадью bounding box
        max_square = 0
        car = {}
        for one in cars:
            detected.remove(one)
            square = (one['xmax'] - one['xmin']) * (one['ymax'] - one['ymin'])

            if square > max_square:
                car = one
                max_square = square

        crossed_objects = [car, ]
        for obj in detected:
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

        self.get_lidar_points()
        lidar = list(self.lidar_points)

        obj_inside = []
        for obj in crossed_objects:
            angles = self.calculate_points_angles_objects(obj)
            lidar_points = self.calculate_lidar_points_inside(angles, lidar)

            points = np.array(lidar_points['in'])
            points = np.delete(points, 3, axis=1)

            obj_inside.append(points)


if __name__ == '__main__':
    rospy.init_node("rl_client", anonymous=True)

    project_path = Path(__file__).resolve().parents[3]
    config_path = project_path.joinpath('config/config.conf')
    config = ConfigFactory.parse_file(config_path)

    env = arctic_env_maker(config.rl_agent.debug_config)

    det = env.get_ssd_lead_information()
    env.calculate_car_distance_v1(det)
