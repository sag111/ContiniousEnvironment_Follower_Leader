from math import pi, degrees, radians, cos, sin, atan, acos, asin, sqrt
import numpy as np
from scipy.spatial import distance

from utils.misc import angle_correction, rotateVector, calculateAngle


class LaserSensor():
    """Реализует один лазерный сенсор лидара"""

    def __init__(self,
                 host_object,
                 available_angle=360,
                 angle_step=10,  # в градусах
                 discretization_rate=20,  # число пикселей,
                 sensor_range=5,  # в метрах
                 distance_variance=0,
                 angle_variance=0,
                 sensor_speed=0.1,
                 return_all_points=False,
                 add_noise=False
                 ):  # в секундах? Пока не используется

        self.host_object = host_object

        self.available_angle = min(360, available_angle)
        self.angle_step = angle_step

        self.range = sensor_range

        self.distance_variance = distance_variance
        self.angle_variance = angle_variance

        self.sensor_speed = sensor_speed
        self.return_all_points = return_all_points
        self.discretization_rate = discretization_rate
        self.sensed_points = list()

    def scan(self, env):
        """строит поля точек лидара.
           Входные параметры:
           env (Game environment):
               среда, в которой осуществляется сканирование;
            Возвращает:
            sensed_points (list):
                список точек, которые отследил лидар.
            """

        # Если на нужной дистанции нет ни одного объекта - просто рисуем крайние точки, иначе нужно будет идти сложным путём
        objects_in_range = list()

        env_range = self.range * env.PIXELS_TO_METER

        for cur_object in env.game_object_list:
            if cur_object is env.follower:
                continue

            if cur_object.blocks_vision:
                if distance.euclidean(cur_object.position, self.host_object.position) <= env_range:
                    objects_in_range.append(cur_object)

        # Далее определить, в какой стороне находится объект из списка, и если он входит в область лидара, ставить точку как надо
        # иначе -- просто ставим точку на максимуме
        border_angle = int(self.available_angle / 2)

        x1 = self.host_object.position[0]
        y1 = self.host_object.position[1]

        self.sensed_points = list()
        angles = list()

        cur_angle_diff = 0

        angles.append(self.host_object.direction)

        while cur_angle_diff < border_angle:
            cur_angle_diff += self.angle_step

            angles.append(angle_correction(self.host_object.direction + cur_angle_diff))
            angles.append(angle_correction(self.host_object.direction - cur_angle_diff))

        for angle in angles:

            x2, y2 = (x1 + env_range * cos(radians(angle)), y1 - env_range * sin(radians(angle)))

            point_to_add = None
            object_in_sight = False

            for i in range(0, self.discretization_rate):
                u = i / self.discretization_rate
                cur_point = ((x2 * u + x1 * (1 - u)), (y2 * u + y1 * (1 - u)))

                if self.return_all_points:
                    self.sensed_points.append(cur_point)

                for cur_object in objects_in_range:
                    if cur_object.rectangle.collidepoint(cur_point):
                        point_to_add = np.array(cur_point, dtype=np.float32)
                        object_in_sight = True
                        break

                if object_in_sight:
                    break

            if point_to_add is None:
                point_to_add = np.array((x2, y2), dtype=np.float32)

            if not self.return_all_points:
                self.sensed_points.append(point_to_add)

        return self.sensed_points

    #         def show(self, display):
    #             pass

    @staticmethod
    def _add_noise(val, variance):
        return max(np.random.normal(val, variance), 0)

class ObservedLeaderPositions_packmanStyle:
    """
    Класс, отслеживающий наблюдаемые позиции лидера.
    Класс генерирует наблюдения двух типов:
        вектора до определённых позиций
        радар, указывающий, есть ли позиции лидера в определённых секторах полукруга перед преследователем
    отслеживать можно самые новые позиции лидера или самые старые
    TODO: Добавить вариант отслеживания позиций или радара до ближайших точек до преследователя
    """
    def __init__(self,
                 host_object,
                 position_sequence_length=100,
                 radar_sectors_number=180):
        """
        :param host_object: робот пресследователь, на котором работает этот сенсор
        :param position_sequence_length: длина последовательности, которая будет использоваться радаром
        :param radar_sectors_number: количество секторов в радаре
        """
        self.host_object = host_object
        self.position_sequence_length = position_sequence_length
        self.radar_sectors_number = radar_sectors_number
        self.sectorsAngle = np.pi / radar_sectors_number
        self.leader_positions_hist = list()
        self.radar_values = None
        self.vecs_values = None

    def get_vectors_to_position(self, positions_time_mode="new"):
        vecs_follower_to_leadhistory_far = np.zeros((self.position_sequence_length, 2))
        if len(self.leader_positions_hist) > 0:
            if positions_time_mode=="new":
                vecs = np.array(self.leader_positions_hist[-self.position_sequence_length:]) - self.host_object.position
            elif positions_time_mode=="old":
                vecs = np.array(self.leader_positions_hist[:self.position_sequence_length]) - self.host_object.position
            vecs_follower_to_leadhistory_far[:min(len(self.leader_positions_hist), self.position_sequence_length)] = vecs
        self.vecs_values = vecs_follower_to_leadhistory_far
        return vecs_follower_to_leadhistory_far

    def get_radar_values(self, positions_time_mode="old"):
        followerDirVec = rotateVector(np.array([1, 0]), self.host_object.direction)
        followerRightDir = self.host_object.direction + 90
        if followerRightDir >= 360:
            followerRightDir -= 360
        followerRightVec = rotateVector(np.array([1, 0]), followerRightDir)
        """
        distances_follower_to_leadhistory = np.linalg.norm(vecs_follower_to_leadhistory, axis=1)
        angles_history_to_dir = calculateAngle(np.array([self.leader.position-self.follower.position, self.leader.position, self.follower.position]), followerDirVec)
        angles_history_to_right = calculateAngle(np.array([self.leader.position-self.follower.position, self.leader.position, self.follower.position]), followerRightVec)
        """
        radar_values = np.zeros(self.radar_sectors_number)
        if len(self.leader_positions_hist) > 0:
            if positions_time_mode=="new":
                chosen_dots = np.array(self.leader_positions_hist[-self.position_sequence_length:])
            elif positions_time_mode=="old":
                chosen_dots = np.array(self.leader_positions_hist[:self.position_sequence_length])
            vecs_follower_to_leadhistory = chosen_dots - self.host_object.position
            distances_follower_to_chosenDots = np.linalg.norm(vecs_follower_to_leadhistory, axis=1)
            angles_history_to_dir = calculateAngle(vecs_follower_to_leadhistory, followerDirVec)
            angles_history_to_right = calculateAngle(vecs_follower_to_leadhistory, followerRightVec)
            angles_history_to_right[angles_history_to_dir > np.pi / 2] = -angles_history_to_right[
                angles_history_to_dir > np.pi / 2]
            for i in range(self.radar_sectors_number):
                secrot_dots_distances = distances_follower_to_chosenDots[
                    (angles_history_to_right >= self.sectorsAngle * i) & (
                            angles_history_to_right < self.sectorsAngle * (i + 1))]
                if len(secrot_dots_distances) > 0:
                    radar_values[i] = np.min(secrot_dots_distances)
        self.radar_values = radar_values
        return radar_values

    def update_observations_hist(self, leader_position):
        """
        Обновляет историю позиций.
        Добавляет наблюдаемую позицию лидера в историю.
        Удаляет из истории позиции, которые преследователь прошёл
        :param leader_position: np.array (2,) - 2 абсолютные координаты лидера
        :return:
        """
        self.leader_positions_hist.append(leader_position.copy())
        norms = np.linalg.norm(np.array(self.leader_positions_hist) - self.host_object.position, axis=1)
        indexes = np.nonzero(norms <= max(self.host_object.width, self.host_object.height))[0]
        for index in sorted(indexes, reverse=True):
            del self.leader_positions_hist[index]

    def reset(self):
        self.leader_positions_hist = list()


# Можно конечно через getattr из модуля брать, но так можно проверку добавить
SENSOR_NAME_TO_CLASS = {
    "LaserSensor": LaserSensor,
    "ObservedLeaderPositions_packmanStyle": ObservedLeaderPositions_packmanStyle
}
