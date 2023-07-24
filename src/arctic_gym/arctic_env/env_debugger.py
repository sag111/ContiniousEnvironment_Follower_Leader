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

    @staticmethod
    def _calculate_points_angles_objects(camera_object):
        """
        функция вычисления углов по значениям bounding box
        Args:
            camera_object = ['xmin': 120, 'ymin' : 100, 'xmax': 300 , 'ymax':400 ]
        Returns:
            Углы для ориентации объекта
            [p_theta1, p_phi1, p_theta2, p_phi2]
        """
        # camera_object = next((x for x in camera_object if x["name"] == "car"), None)
        xmin = camera_object['xmin']
        ymin = camera_object['ymin']
        xmax = camera_object['xmax']
        ymax = camera_object['ymax']

        # xcent = (xmin + xmax)/2
        # ycent = (ymin + ymax)/2

        xmin = xmin - 20
        xmax = xmax + 20
        # ymin = ycent - 20
        # ymax = ycent + 20

        p_theta1 = atan((2 * xmin - 640) / 640 * tan(80 / 2))
        p_phi1 = atan(-((2 * ymin - 480) / 480) * tan(64 / 2))  # phi
        p_theta2 = atan((2 * xmax - 640) / 640 * tan(80 / 2))  # theta
        p_phi2 = atan(-((2 * ymax - 480) / 480) * tan(64 / 2))  # phi

        angles_object = [p_theta1, p_phi1, p_theta2, p_phi2]

        return angles_object

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

    def calculate_length_to_leader(self, camera_objects):

        """
        Функция определения расстояния до ведущего на основе обработки результата детектирования объектов на изображении
        и облака точек лидара. На основе полученных bounding box происходит сопоставление их с точками лидара, используя
        результат функции calculate_points_angles_objects. В результате, из всего облака точек лидара происходит выделение
        только точек лидара, использую BB и углы из calculate_points_angles_objects.
        Далее, на основе полученной информации берется ближайшая точка, и на основе нее вычисляется расстояние до ведущего.
        Также, из всего облака точек, удаляются точки ведущего и в дальнейшем не учитвваются в обработке препятствий.

        Args:
            camera_objects = np.array()

        Returns:
            Результат:length_to_leader - расстояние до ведущего, other_points - список облака точек без ведущего
            length_to_leader = x
            other_points = list([x, y, z])

        """

        max_dist = 25
        length_to_leader = 50
        object_coord = []
        leader_info = next((x for x in camera_objects if x["name"] == "car"), None)

        other_points = list()

        camera_yaw_state_info = self.sub.get_camera_yaw_state()
        camera_yaw = camera_yaw_state_info.process_value

        # TODO : перепроверить и оптимизировать (в случае, если пробовать вариант с "закрытым" коридором)
        # if i[-1] in [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
        # выделение точек ведущего из всего облака точек
        for i in self.lidar_points:
            if leader_info is not None:
                angles_object = self._calculate_points_angles_objects(leader_info)
                if i[0] ** 2 + i[1] ** 2 <= max_dist ** 2\
                        and (tan(np.deg2rad(-40))+tan(camera_yaw)) * i[0] <= i[1] <= (tan(np.deg2rad(40))+tan(camera_yaw)) * i[0] \
                        and ((tan(angles_object[2])+tan(camera_yaw)) * i[0]) <= i[1] <= ((tan(angles_object[0])+tan(camera_yaw)) * i[0]) \
                        and (tan(angles_object[3]) * i[0]) <= i[2] <= (tan(angles_object[1]) * i[0]):
                    object_coord.append(i)

                    if sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2) <= length_to_leader:
                        length_to_leader = sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2)
                elif i[0] ** 2 + i[1] ** 2 >= 1:
                    other_points.append([i[0], i[1], i[2]])
            elif i[0] ** 2 + i[1] ** 2 >= 1:
                other_points.append([i[0], i[1], i[2]])
                length_to_leader = None

        # print(f'Расстояние до ведущего, определенное с помощью лидара: {length_to_leader}')

        return length_to_leader, other_points


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