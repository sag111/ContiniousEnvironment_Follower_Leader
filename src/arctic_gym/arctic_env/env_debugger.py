import rospy
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import statistics

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
                                        fov: float = 64.0,
                                        scale: int = 20) -> dict:
        """
        Вычисление углов по значениям bounding box, вычисления основано на цилиндрической системе координат,
        центр изображения - центр системы координат

        :param obj: словарь объекта с ключами name - имя объекта; xmin, xmax, ymin, ymax - координаты bounding box
        :param width: ширина изображения в пикселях
        :param height: высота изображения в пикселях
        :param hov: горизонтальный угол обзора камеры
        :param fov: вертикальный угол обзора камеры
        :param scale: расширение границ bounding box по горизонтали

        :return:
            Словарь с граничными углами области объекта по вертикали (phi1, phi2) и горизонтали (theta1, theta2)
        """

        xmin = obj['xmin']
        xmax = obj['xmax']
        ymin = obj['ymin']
        ymax = obj['ymax']

        xmin -= scale
        xmax += scale

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

    def calculate_lidar_points_inside(self, name, angles: dict, lidar: list, maxd: int = 25) -> dict:
        """
        Вычисление точек лидара находящихся внутри и снаружи области ограниченной углами angles

        :param name: имя объекта, если "car" сохраняем точки за пределами bounding box
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
            dist = np.linalg.norm(i[:3])

            k1_x = (tan(np.deg2rad(-40)) + tan(camera_yaw)) * i[0]
            k2_x = (tan(np.deg2rad(40)) + tan(camera_yaw)) * i[0]

            theta2_x = (tan(angles["theta2"]) + tan(camera_yaw)) * i[0]
            theta1_x = (tan(angles["theta1"]) + tan(camera_yaw)) * i[0]

            phi2_x = tan(angles["phi2"]) * i[0]
            phi1_x = tan(angles["phi1"]) * i[0]

            if dist <= maxd and k1_x <= i[1] <= k2_x and theta2_x <= i[1] <= theta1_x and phi2_x <= i[2] <= phi1_x:
                points_inside.append(list(i[:3]))
            elif 2 <= dist <= maxd and name == "car":
                points_outside.append(list(i[:3]))

        return {
            "in": points_inside,
            "out": points_outside
        }

    def calculate_car_distance_v1(self, detected: list):
        """
        Определение расстояния до машины. Выбираем bounding box машины и пересекающиеся с ней объекты, вырезаем точки
        лидара, которые располагаются в области пересекающихся объектов

        :param detected: список распознанных объектов, каждый объект представляет собой словарь со следующими ключами
            name - имя объекта; xmin, xmax, ymin, ymax - координаты bounding box
        """
        cars = [x for x in detected if x['name'] == "car"]

        # если нет машины передаем все точки лидара как препятствия
        if cars == []:
            self.get_lidar_points()
            lidar = list(self.lidar_points)
            points_outside = []
            for i in lidar:
                dist = np.linalg.norm(i[:3])
                if 2 <= dist <= 40:
                    points_outside.append(i[:3])

            return None, points_outside

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

        # считаем точки лидара внутри и снаружи выделенных bounding box
        obj_inside = []
        outside_car = []
        for obj in crossed_objects:
            angles = self.calculate_points_angles_objects(obj)
            lidar_points = self.calculate_lidar_points_inside(obj["name"], angles, lidar)

            points = np.array(lidar_points['in'])

            if obj["name"] == "car":
                outside_car = np.array(lidar_points['out'])

            if points != []:
                norms = np.linalg.norm(points, axis=1)
                obj_inside.append({"name": obj["name"], "data": dict(zip(norms, points))})

        # удаляем точки лидара других объектов, которые пересекаются с bounding box машины
        car_data = obj_inside[0]["data"]
        outside_car_bb = {}
        for other_data in obj_inside[1:]:
            car_keys = np.array(list(car_data.keys()))
            other_keys = np.array(list(other_data["data"].keys()))

            intersections = np.intersect1d(car_keys, other_keys)
            for inter in intersections:
                outside_car_bb[inter] = car_data.pop(inter)

        # по гистограмме определяем расстояние, которое встречается чаще остальных
        count, distance, _ = plt.hist(car_data.keys())
        idx = np.where(count == max(count))
        best_distance = distance[idx]

        # выбираем расстояния, удовлетворяющие диапазону
        outside_car_dd = {}
        state_data = car_data.copy().keys()
        for pt in state_data:
            if best_distance - 3 <= pt <= best_distance + 3:
                pass
            else:
                outside_car_dd[pt] = car_data.pop(pt)

        # объединяем все точки за границей bounding box машины
        points_outside = np.concatenate((
            outside_car,
            np.array(list(outside_car_bb.values())),
            np.array(list(outside_car_dd.values()))
        ))

        final_distance = np.array(list(car_data.keys())).mean()

        return final_distance, points_outside


if __name__ == '__main__':
    rospy.init_node("rl_client", anonymous=True)

    project_path = Path(__file__).resolve().parents[3]
    config_path = project_path.joinpath('config/config.conf')
    config = ConfigFactory.parse_file(config_path)

    env = arctic_env_maker(config.rl_agent.debug_config)

    det = env.get_ssd_lead_information()
    env.calculate_car_distance_v1(det)
