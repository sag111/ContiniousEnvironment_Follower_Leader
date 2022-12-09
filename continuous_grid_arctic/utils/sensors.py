from math import pi, degrees, radians, cos, sin, atan, acos, asin, sqrt
import numpy as np
import pygame
from scipy.spatial import distance
from collections import deque
import itertools

try:
    from utils.misc import angle_correction, rotateVector, calculateAngle, distance_to_rect
except:
    from continuous_grid_arctic.utils.misc import angle_correction, rotateVector, calculateAngle, distance_to_rect


class LaserSensor():
    """Реализует лазерный лидар"""

    def __init__(self,
                 host_object,
                 sensor_name,
                 available_angle=360,
                 angle_step=10,  # в градусах
                 points_number=20,  # число пикселей,
                 sensor_range=5,  # в метрах
                 distance_variance=0,
                 angle_variance=0,
                 sensor_speed=0.1,
                 return_all_points=False,
                 add_noise=False,
                 return_only_distances=False
                 ):  # в секундах? Пока не используется

        self.host_object = host_object
        self.sensor_name = sensor_name
        self.return_only_distances = return_only_distances

        self.available_angle = min(360, available_angle)
        self.angle_step = angle_step

        self.range = sensor_range

        self.distance_variance = distance_variance
        self.angle_variance = angle_variance

        self.sensor_speed = sensor_speed
        self.return_all_points = return_all_points
        self.sensed_points = list()
        self.points_number = points_number
        self.data_shape = int(self.available_angle / self.angle_step)
        if self.return_all_points:
            self.data_shape = self.data_shape * points_number  # + 1

    def __len__(self):
        return self.data_shape

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

        for cur_object in (env.game_object_list + env.game_dynamic_list):
            if cur_object is env.follower:
                continue

            if cur_object.blocks_vision:
                if distance_to_rect(self.host_object.position, cur_object) <= env_range + (3 * env.PIXELS_TO_METER):
                    objects_in_range.append(cur_object)

        # Далее определить, в какой стороне находится объект из списка, и если он входит в область лидара, ставить точку как надо
        # иначе -- просто ставим точку на максимуме
        border_angle = int(self.available_angle / 2)

        x1 = self.host_object.position[0]
        y1 = self.host_object.position[1]

        self.sensed_points = list()
        angles = list()

        cur_angle_diff = 0

        angles.append(-self.host_object.direction)

        while cur_angle_diff < border_angle:
            cur_angle_diff += self.angle_step

            angles.append(angle_correction(-self.host_object.direction + cur_angle_diff))
            angles.append(angle_correction(-self.host_object.direction - cur_angle_diff))

        for angle in angles:

            x2, y2 = (x1 + env_range * cos(radians(angle)), y1 - env_range * sin(radians(angle)))

            point_to_add = None
            object_in_sight = False

            for i in range(0, self.points_number):
                u = i / self.points_number
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

        if self.return_only_distances:
            return np.linalg.norm(self.sensed_points - self.host_object.position, axis=1)
        else:
            return self.sensed_points - self.host_object.position

    def show(self, env):
        for cur_point in self.sensed_points:
            pygame.draw.circle(env.gameDisplay, (250, 50, 50), cur_point, 3)

    # @staticmethod
    # def _add_noise(val, variance):
    #    return max(np.random.normal(val, variance), 0)


class LeaderPositionsTracker:
    """
        Класс, отслеживающий наблюдаемые позиции лидера.
        не генерирует наблюдения, но хранит историю позиций лидера для других сенсоров.
        TODO: Может имеет смысл переделать на относительные координаты, это же ведомый отслеживает относительно себя, но тогда другие сенсоры тоже надо фиксить.
    """

    def __init__(self,
                 host_object,
                 sensor_name,
                 eat_close_points=True,
                 max_point=5000,
                 saving_period=5,
                 generate_corridor=True,
                 start_corridor_behind_follower=False
                 ):
        self.sensor_name = sensor_name
        self.host_object = host_object
        self.max_point = max_point
        self.eat_close_points = eat_close_points
        # TODO: попробовать реализовать как ndarray, может быстрее будет, потому что другие сенсоры это как ndarray используют
        self.leader_positions_hist = deque()
        self.saving_period = saving_period
        self.saving_counter = 0
        self.generate_corridor = generate_corridor
        self.corridor = deque()
        self.right_border_dot = np.array([0, 0])
        self.left_border_dot = np.array([0, 0])
        self.start_corridor_behind_follower = start_corridor_behind_follower

    def scan(self, env):
        # если сам сенсор отслеживает перемещение
        if self.saving_counter % self.saving_period == 0:
            # Если позиция лидера не изменилась с последнего обсерва, просто возвращаем, что есть, ничего не обновляем
            if len(self.leader_positions_hist) > 0 and (self.leader_positions_hist[-1] == env.leader.position).all():
                if self.generate_corridor:
                    return self.leader_positions_hist, self.corridor
                else:
                    return self.leader_positions_hist
            # Если симуляция только началась, сохраняем текущую ведомого, чтоб начать от неё строить коридор
            if len(self.leader_positions_hist) == 0:
                self.leader_positions_hist.append(self.host_object.position.copy())
            self.leader_positions_hist.append(env.leader.position.copy())
            if self.generate_corridor and len(self.leader_positions_hist) > 1:
                last_2points_vec = self.leader_positions_hist[-1] - self.leader_positions_hist[-2]
                last_2points_vec *= env.max_dev / np.linalg.norm(last_2points_vec)
                right_border_dot = rotateVector(last_2points_vec, 90)
                right_border_dot += self.leader_positions_hist[-2]
                left_border_dot = rotateVector(last_2points_vec, -90)
                left_border_dot += self.leader_positions_hist[-2]
                self.corridor.append([right_border_dot, left_border_dot])
        # Можно ещё брать из среды, но там частота сохранения другая
        # self.leader_positions_hist = env.leader_factual_trajectory[::self.saving_period]
        # assert env.frames_per_step % env.trajectory_saving_period == 0
        # print(self.leader_positions_hist)
        # print(env.leader_factual_trajectory[::self.saving_period*(int(env.frames_per_step / env.trajectory_saving_period))])

        self.saving_counter += 1

        if self.eat_close_points and len(self.leader_positions_hist) > 0:
            norms = np.linalg.norm(np.array(self.leader_positions_hist) - self.host_object.position, axis=1)
            indexes = np.nonzero(norms <= max(self.host_object.width, self.host_object.height))[0]
            for index in sorted(indexes, reverse=True):
                del self.leader_positions_hist[index]
        if self.generate_corridor:
            return self.leader_positions_hist, self.corridor
        else:
            return self.leader_positions_hist

    def reset(self):
        self.leader_positions_hist.clear()
        self.corridor.clear()
        self.saving_counter = 0

    def show(self, env):
        for point in self.leader_positions_hist:
            pygame.draw.circle(env.gameDisplay, (50, 10, 10), point, 3)

        if len(self.corridor) > 1:
            pygame.draw.lines(env.gameDisplay, (150, 120, 50), False, [x[0] for x in self.corridor], 3)
            pygame.draw.lines(env.gameDisplay, (150, 120, 50), False, [x[1] for x in self.corridor], 3)
        pass


class LeaderPositionsTracker_v2(LeaderPositionsTracker):
    def scan(self, env):
        # если сам сенсор отслеживает перемещение
        if self.saving_counter % self.saving_period == 0:
            # Если позиция лидера не изменилась с последнего обсерва, просто возвращаем, что есть, ничего не обновляем
            if len(self.leader_positions_hist) > 0 and (self.leader_positions_hist[-1] == env.leader.position).all():
                if self.generate_corridor:
                    return self.leader_positions_hist, self.corridor
                else:
                    return self.leader_positions_hist
            # Если симуляция только началась, сохраняем текущую ведомого, чтоб начать от неё строить коридор
            if len(self.leader_positions_hist) == 0 and self.saving_counter == 0:
                # self.leader_positions_hist.append(self.host_object.position.copy())
                # TODO ; точка позиции за ведомым (для отстроения коридора)
                # TODO : вариант с отсроением коридора от точки за метр от ведомого
                if self.start_corridor_behind_follower:
                    point_start_distance_behind_follower = 50
                    point_start_position_theta = angle_correction(self.host_object.direction + 180)
                    point_behind_follower = np.array(
                        (point_start_distance_behind_follower * cos(radians(point_start_position_theta)),
                         point_start_distance_behind_follower * sin(radians(point_start_position_theta)))) \
                                              + self.host_object.position
                    first_dots_for_follower_count = int(
                        distance.euclidean(point_behind_follower, env.leader.position) / (
                                self.saving_period * 5 * env.leader.max_speed))

                    self.leader_positions_hist.extend(np.array(x) for x in
                                                      zip(np.linspace(point_behind_follower[0], env.leader.position[0],
                                                                      first_dots_for_follower_count),
                                                          np.linspace(point_behind_follower[1], env.leader.position[1],
                                                                      first_dots_for_follower_count)))
                # TODO : вариант с отсроением коридора от точки ведомого
                else:
                    first_dots_for_follower_count = int(
                        distance.euclidean(self.host_object.position, env.leader.position) / (
                                self.saving_period * 5 * env.leader.max_speed))
                    self.leader_positions_hist.extend(np.array(x) for x in
                                                      zip(np.linspace(self.host_object.position[0], env.leader.position[0],
                                                                      first_dots_for_follower_count),
                                                          np.linspace(self.host_object.position[1], env.leader.position[1],
                                                                      first_dots_for_follower_count)))
            else:
                self.leader_positions_hist.append(env.leader.position.copy())

            dists = np.linalg.norm(np.array(self.leader_positions_hist)[:-1, :] -
                                   np.array(self.leader_positions_hist)[1:, :], axis=1)
            path_length = np.sum(dists)
            # while path_length > env.max_distance:
            while path_length > env.corridor_length:
                self.leader_positions_hist.popleft()
                self.corridor.popleft()
                dists = np.linalg.norm(np.array(self.leader_positions_hist)[:-1, :] -
                                       np.array(self.leader_positions_hist)[1:, :], axis=1)
                path_length = np.sum(dists)

            if self.generate_corridor and len(self.leader_positions_hist) > 1:
                if self.saving_counter == 0:
                    for i in range(len(self.leader_positions_hist) - 1, 0, -1):
                        last_2points_vec = self.leader_positions_hist[i] - self.leader_positions_hist[i-1]
                        # last_2points_vec *= env.max_dev / np.linalg.norm(last_2points_vec)
                        last_2points_vec *= env.corridor_width / np.linalg.norm(last_2points_vec)
                        right_border_dot = rotateVector(last_2points_vec, 90)
                        right_border_dot += self.leader_positions_hist[-i-1]
                        left_border_dot = rotateVector(last_2points_vec, -90)
                        left_border_dot += self.leader_positions_hist[-i-1]
                        self.corridor.append([right_border_dot, left_border_dot])
                last_2points_vec = self.leader_positions_hist[-1] - self.leader_positions_hist[-2]
                # last_2points_vec *= env.max_dev / np.linalg.norm(last_2points_vec)
                last_2points_vec *= env.corridor_width / np.linalg.norm(last_2points_vec)
                right_border_dot = rotateVector(last_2points_vec, 90)
                right_border_dot += self.leader_positions_hist[-2]
                left_border_dot = rotateVector(last_2points_vec, -90)
                left_border_dot += self.leader_positions_hist[-2]
                self.corridor.append([right_border_dot, left_border_dot])
        # Можно ещё брать из среды, но там частота сохранения другая
        # self.leader_positions_hist = env.leader_factual_trajectory[::self.saving_period]
        # assert env.frames_per_step % env.trajectory_saving_period == 0
        # print(self.leader_positions_hist)
        # print(env.leader_factual_trajectory[::self.saving_period*(int(env.frames_per_step / env.trajectory_saving_period))])
        self.saving_counter += 1
        if self.generate_corridor:
            return self.leader_positions_hist, self.corridor
        else:
            return self.leader_positions_hist

    def show(self, env):
        for point in self.leader_positions_hist:
            pygame.draw.circle(env.gameDisplay, (80, 10, 10), point, 3)

        if len(self.corridor) > 1:
            pygame.draw.lines(env.gameDisplay, (150, 120, 50), False, [x[0] for x in self.corridor], 3)
            pygame.draw.lines(env.gameDisplay, (150, 120, 50), False, [x[1] for x in self.corridor], 3)
            pygame.draw.line(env.gameDisplay, (150, 120, 50), self.corridor[0][0], self.corridor[0][1], 3)
            pygame.draw.line(env.gameDisplay, (150, 120, 50), self.corridor[-1][0], self.corridor[-1][1], 3)
        pass


class LeaderTrackDetector_vector:
    """
    Класс, реагирующий на старые позиции лидера и генерирующий вектора до определённых позиций.
    отслеживать можно самые новые позиции лидера или самые старые
    TODO: Добавить вариант отслеживания позиций или радара до ближайших точек до преследователя
    """

    def __init__(self,
                 host_object,
                 sensor_name,
                 position_sequence_length=100,
                 detectable_positions="new"):
        """
        :param host_object: робот пресследователь, на котором работает этот сенсор
        :param position_sequence_length: длина последовательности, которая будет использоваться радаром
        """
        self.sensor_name = sensor_name
        self.host_object = host_object
        self.position_sequence_length = position_sequence_length
        self.vecs_values = np.zeros((self.position_sequence_length, 2), dtype=np.float32)
        self.detectable_positions = detectable_positions

    def scan(self, env, leader_positions_hist):
        self.vecs_values = np.zeros((self.position_sequence_length, 2), dtype=np.float32)
        if len(leader_positions_hist) > 0:
            if self.detectable_positions == "new":
                # vecs = np.array(leader_positions_hist[-self.position_sequence_length:]) - self.host_object.position
                slice = list(itertools.islice(leader_positions_hist, max(0, len(leader_positions_hist) -
                                                                         self.position_sequence_length),
                                              len(leader_positions_hist)))
                vecs = np.array(slice) - self.host_object.position
            elif self.detectable_positions == "old":
                # vecs = np.array(leader_positions_hist[:self.position_sequence_length]) - self.host_object.position
                slice = list(itertools.islice(leader_positions_hist, 0, self.position_sequence_length))
                vecs = np.array(slice) - self.host_object.position
            self.vecs_values[
            :min(len(leader_positions_hist), self.position_sequence_length)] = vecs
        return self.vecs_values

    def show(self, env):
        for i in range(self.vecs_values.shape[0]):
            if np.sum(self.host_object.position + self.vecs_values[i]) > 0:
                # pygame.draw.line(self.gameDisplay, (250, 200, 150), self.follower.position, \
                # self.follower.position+self.follower.sensors["ObservedLeaderPositions_packmanStyle"].vecs_values[i])
                pygame.draw.circle(env.gameDisplay, (255, 100, 50), self.host_object.position +
                                   self.vecs_values[i], 1)

    def reset(self):
        self.vecs_values = np.zeros((self.position_sequence_length, 2), dtype=np.float32)


class LeaderTrackDetector_radar:
    """
    Радар, реагирующий на старые позиции лидера, и указывающий, есть ли позиции лидера в
    определённых секторах полукруга перед преследователем
    отслеживать можно самые новые позиции лидера или самые старые
    TODO: Добавить вариант отслеживания позиций или радара до ближайших точек до преследователя
    """

    def __init__(self,
                 host_object,
                 sensor_name,
                 position_sequence_length=100,
                 detectable_positions="old",
                 radar_sectors_number=180):
        """
        :param host_object: робот пресследователь, на котором работает этот сенсор
        :param position_sequence_length: длина последовательности, которая будет использоваться радаром
        :param radar_sectors_number: количество секторов в радаре
        """
        self.sensor_name = sensor_name
        self.host_object = host_object
        self.detectable_positions = detectable_positions
        self.position_sequence_length = position_sequence_length
        self.radar_sectors_number = radar_sectors_number
        self.sectorsAngle_rad = np.pi / radar_sectors_number
        self.sectorsAngle_deg = 180 / radar_sectors_number
        self.radar_values = np.zeros(self.radar_sectors_number, dtype=np.float32)

    def scan(self, env, leader_positions_hist):
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
        self.radar_values = np.zeros(self.radar_sectors_number, dtype=np.float32)
        if len(leader_positions_hist) > 0:

            if self.detectable_positions == "near":
                leader_positions_hist = np.array(leader_positions_hist)
                vecs_follower_to_leadhistory = leader_positions_hist - self.host_object.position
                distances_follower_to_chosenDots = np.linalg.norm(vecs_follower_to_leadhistory, axis=1)
                closest_indexes = np.argsort(distances_follower_to_chosenDots)
                vecs_follower_to_leadhistory = vecs_follower_to_leadhistory[closest_indexes]
                distances_follower_to_chosenDots = distances_follower_to_chosenDots[closest_indexes]
            else:
                if self.detectable_positions == "new":
                    # chosen_dots = np.array(leader_positions_hist[-self.position_sequence_length:])
                    slice = list(itertools.islice(leader_positions_hist, max(0, len(leader_positions_hist) -
                                                                             self.position_sequence_length),
                                                  len(leader_positions_hist)))
                    chosen_dots = np.array(slice)
                elif self.detectable_positions == "old":
                    # chosen_dots = np.array(leader_positions_hist[:self.position_sequence_length])
                    slice = list(itertools.islice(leader_positions_hist, 0, self.position_sequence_length))
                    chosen_dots = np.array(slice)
                vecs_follower_to_leadhistory = chosen_dots - self.host_object.position
                distances_follower_to_chosenDots = np.linalg.norm(vecs_follower_to_leadhistory, axis=1)
            angles_history_to_dir = calculateAngle(vecs_follower_to_leadhistory, followerDirVec)
            angles_history_to_right = calculateAngle(vecs_follower_to_leadhistory, followerRightVec)
            angles_history_to_right[angles_history_to_dir > np.pi / 2] = -angles_history_to_right[
                angles_history_to_dir > np.pi / 2]
            for i in range(self.radar_sectors_number):
                sector_dots_distances = distances_follower_to_chosenDots[
                    (angles_history_to_right >= self.sectorsAngle_rad * i) & (
                            angles_history_to_right < self.sectorsAngle_rad * (i + 1))]
                if len(sector_dots_distances) > 0:
                    self.radar_values[i] = np.min(sector_dots_distances)
        return self.radar_values

    def reset(self):
        self.radar_values = np.zeros(self.radar_sectors_number, dtype=np.float32)

    def show(self, env):
        for i in range(self.radar_values.shape[0]):
            followerRightDir = self.host_object.direction + 90
            if followerRightDir >= 360:
                followerRightDir -= 360

            for i in range(self.radar_sectors_number):
                if self.radar_values[i] == 0:
                    continue
                followerRightVec = rotateVector(np.array([self.radar_values[i], 0]), followerRightDir)
                relativeDot = rotateVector(followerRightVec, self.sectorsAngle_deg * (self.radar_sectors_number - i) - (
                        self.sectorsAngle_deg / 2))
                absDot = self.host_object.position - relativeDot
                pygame.draw.line(env.gameDisplay, (180, 80, 255), self.host_object.position, absDot, 3)
                # pygame.draw.circle(env.gameDisplay, (255, 80, 180), absDot, 4)


class GreenBoxBorderSensor(LaserSensor):
    """Лидар для отслеживания границ зелёной зоны, в которой должен быть ведущий."""

    def __init__(self, host_object, **kwargs):
        raise ValueError("Для использования нужно раскомментировать в среде вызов self._get_green_zone_border_points("
                         "). Закомментировал, потому что замедляет симмуляцию")
        self.sensor_name = 'GreenBox_Border_Sensor'

        super().__init__(host_object, self.sensor_name, **kwargs)

    def scan(self, env):
        """строит поля точек лидара.
           Входные параметры:
           env (Game environment):
               среда, в которой осуществляется сканирование;
            Возвращает:
            sensed_points (list):
                список точек, которые отследил лидар.
            """

        # Далее определить, в какой стороне находится объект из списка, и если он входит в область лидара, ставить точку как надо
        # иначе -- просто ставим точку на максимуме

        env_range = self.range * env.PIXELS_TO_METER

        border_angle = int(self.available_angle / 2)

        x1 = self.host_object.position[0]
        y1 = self.host_object.position[1]

        self.sensed_points = list()
        angles = list()

        cur_angle_diff = 0

        angles.append(-self.host_object.direction)

        while cur_angle_diff < border_angle:
            cur_angle_diff += self.angle_step

            angles.append(angle_correction(-self.host_object.direction + cur_angle_diff))
            angles.append(angle_correction(-self.host_object.direction - cur_angle_diff))

        list_points_to_scan = env.left_border_points_list + env.right_border_points_list

        for angle in angles:

            x2, y2 = (x1 + env_range * cos(radians(angle)), y1 - env_range * sin(radians(angle)))

            point_to_add = None
            object_in_sight = False

            for i in range(0, self.points_number):
                u = i / self.points_number
                cur_point = ((x2 * u + x1 * (1 - u)), (y2 * u + y1 * (1 - u)))

                if self.return_all_points:
                    self.sensed_points.append(cur_point)

                for cur_point_to_scan in list_points_to_scan:
                    if distance.euclidean(cur_point_to_scan, cur_point) <= 5:
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

    def show(self, env):
        for cur_point in self.sensed_points:
            pygame.draw.circle(env.gameDisplay, env.colours["blue"], cur_point, 2)


class LeaderCorridor_lasers:
    def __init__(self,
                 host_object,
                 sensor_name,
                 react_to_safe_corridor=True,
                 react_to_obstacles=False,
                 react_to_green_zone=False,
                 front_lasers_count=3,
                 back_lasers_count=0,
                 laser_length=100):
        """

        :param host_object: робот, на котором висит сенсор
        :param sensor_name: название сенсора. Важно, если  несколько одинаковых
        :param react_to_obstacles: должны ли лазеры реагировать на препятствия
        :param react_to_green_zone: должны ли лазеры реагировать на переднюю из заднюю границы зеленой зоны
        """
        self.host_object = host_object
        self.sensor_name = sensor_name
        # TODO: сделать гибкую настройку лазеров
        # assert front_lasers_count in [3, 5]
        # assert back_lasers_count in [0, 2]
        self.front_lasers_count = front_lasers_count
        self.back_lasers_count = back_lasers_count
        self.laser_length = laser_length
        self.lasers_end_points = []
        self.lasers_collides = []
        self.react_to_safe_corridor = react_to_safe_corridor
        self.react_to_obstacles = react_to_obstacles
        self.react_to_green_zone = react_to_green_zone

    def ccw(A, B, C):
        return (C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0]) > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])

    # Return true if line segments AB and CD intersect
    def intersect(A, B, C, D):
        return np.logical_and(LeaderCorridor_lasers.ccw(A, C, D) != LeaderCorridor_lasers.ccw(B, C, D),
                              LeaderCorridor_lasers.ccw(A, B, C) != LeaderCorridor_lasers.ccw(A, B, D))

    def perp(a):
        # https://stackoverflow.com/a/3252222/4807259
        b = np.empty_like(a)
        b[:, 0] = -a[:, 1]
        b[:, 1] = a[:, 0]
        return b

    # line segment a given by endpoints a1, a2
    # line segment b given by endpoints b1, b2
    # return
    def seg_intersect(a1, a2, b1, b2):
        # https://stackoverflow.com/a/3252222/4807259
        # ДОБАВИТЬ ДЛЯ КОЛЛИНЕАРНОСТИ  УСЛОВИЕ, ЧТОБ УКАЗЫВАТЬ БЛИЖАЙШИЙ КОНЕЦ КАК ТОЧКУ ПЕРЕСЕЧЕНИЯ
        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = LeaderCorridor_lasers.perp(da)
        denom = np.dot(dap, db.transpose())

        # num = np.zeros(dp.shape[0])
        # а можно ли как-то без цикла?
        # for i in range(dp.shape[0]):
        # num[i] = dot( dap[i,:], dp[i,:] )
        num = np.sum(np.multiply(dap, dp), axis=1)
        return (num[:, np.newaxis] / denom) * db + b1

    def scan(self, env, corridor):
        self.lasers_collides = []
        self.lasers_end_points = []
        self.lasers_end_points.append(
            self.host_object.position + rotateVector(np.array([self.laser_length, 0]), self.host_object.direction - 40))
        self.lasers_end_points.append(
            self.host_object.position + rotateVector(np.array([self.laser_length, 0]), self.host_object.direction))
        self.lasers_end_points.append(
            self.host_object.position + rotateVector(np.array([self.laser_length, 0]), self.host_object.direction + 40))
        if self.front_lasers_count == 5:
            self.lasers_end_points.append(
                self.host_object.position + rotateVector(np.array([self.laser_length, 0]),
                                                         self.host_object.direction - 90))
            self.lasers_end_points.append(
                self.host_object.position + rotateVector(np.array([self.laser_length, 0]),
                                                         self.host_object.direction + 90))
        if self.back_lasers_count == 2:
            self.lasers_end_points.append(
                self.host_object.position + rotateVector(np.array([self.laser_length, 0]),
                                                         self.host_object.direction - 150))
            self.lasers_end_points.append(
                self.host_object.position + rotateVector(np.array([self.laser_length, 0]),
                                                         self.host_object.direction + 150))
        if len(corridor) > 1:
            corridor_lines = list()
            if self.react_to_safe_corridor:
                for i in range(len(corridor) - 1):
                    corridor_lines.append([corridor[i][0], corridor[i + 1][0]])
                    corridor_lines.append([corridor[i][1], corridor[i + 1][1]])
            if self.react_to_green_zone:
                corridor_lines.append([corridor[0][0], corridor[0][1]])
                corridor_lines.append([corridor[-1][0], corridor[-1][1]])
            if self.react_to_obstacles:
                # TODO : проверка списка динам препятствий
                for cur_object in (env.game_object_list + env.game_dynamic_list):
                    if cur_object is env.follower:
                        continue
                    if cur_object.blocks_vision:
                        corridor_lines.append([cur_object.rectangle.bottomleft, cur_object.rectangle.bottomright])
                        corridor_lines.append([cur_object.rectangle.topright, cur_object.rectangle.bottomright])
                        corridor_lines.append([cur_object.rectangle.topright, cur_object.rectangle.topleft])
                        corridor_lines.append([cur_object.rectangle.bottomleft, cur_object.rectangle.topleft])
            corridor_lines = np.array(corridor_lines, dtype=np.float32)
            lasers_values = []
            self.lasers_collides = []
            for laser_end_point in self.lasers_end_points:
                # эта функция не работает с коллинеарными
                # есть вариант лучше, но медленней
                # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
                rez = LeaderCorridor_lasers.intersect(corridor_lines[:, 0, :], corridor_lines[:, 1, :],
                                                      np.array([self.host_object.position]),
                                                      np.array([laser_end_point]))
                intersected_line = corridor_lines[rez]
                if len(intersected_line) > 0:
                    x = LeaderCorridor_lasers.seg_intersect(intersected_line[:, 0, :], intersected_line[:, 1, :],
                                                            np.array([self.host_object.position]),
                                                            np.array([laser_end_point]))
                    # TODO: исключить коллинеарные, вместо их точек пересечения добавить ближайшую точку коллинеарной границы
                    # но это бесполезно при использовании функции intersect, которая не работает с коллинеарными
                    exclude_rows = np.concatenate([np.nonzero(np.isinf(x))[0], np.nonzero(np.isnan(x))[0]])
                    norms = np.linalg.norm(x - self.host_object.position, axis=1)
                    lasers_values.append(np.min(norms))
                    closest_dot_idx = np.argmin(np.linalg.norm(x - self.host_object.position, axis=1))
                    self.lasers_collides.append(x[closest_dot_idx])
                else:
                    self.lasers_collides.append(laser_end_point)
        obs = np.ones(self.front_lasers_count + self.back_lasers_count, dtype=np.float32) * self.laser_length
        for i, collide in enumerate(self.lasers_collides):
            obs[i] = np.linalg.norm(collide - self.host_object.position)
        return obs

    def show(self, env):
        for laser_end_point in self.lasers_end_points:
            pygame.draw.line(env.gameDisplay, (200, 0, 100), self.host_object.position, laser_end_point)

        for laser_collide in self.lasers_collides:
            pygame.draw.circle(env.gameDisplay, (200, 0, 100), laser_collide, 5)

class LeaderCorridor_lasers_v2(LeaderCorridor_lasers):

    def scan(self, env, corridor):
        self.count_lasers = 12
        self.lasers_collides = []
        self.lasers_end_points = []
        print("!!!!!!!!!!!!!!!!!!!!!!!",self.front_lasers_count+self.back_lasers_count)
        if self.front_lasers_count+self.back_lasers_count == self.count_lasers:
            for i in range(self.count_lasers):
                self.lasers_end_points.append(self.host_object.position + rotateVector(np.array([self.laser_length, 0]),
                                                               self.host_object.direction + i*30))

        if len(corridor) > 1:
            corridor_lines = list()
            if self.react_to_safe_corridor:
                for i in range(len(corridor) - 1):
                    corridor_lines.append([corridor[i][0], corridor[i + 1][0]])
                    corridor_lines.append([corridor[i][1], corridor[i + 1][1]])
            if self.react_to_green_zone:
                corridor_lines.append([corridor[0][0], corridor[0][1]])
                corridor_lines.append([corridor[-1][0], corridor[-1][1]])
            if self.react_to_obstacles:
                # TODO : проверка списка динам препятствий
                for cur_object in (env.game_object_list + env.game_dynamic_list):
                    if cur_object is env.follower:
                        continue
                    if cur_object.blocks_vision:
                        corridor_lines.append([cur_object.rectangle.bottomleft, cur_object.rectangle.bottomright])
                        corridor_lines.append([cur_object.rectangle.topright, cur_object.rectangle.bottomright])
                        corridor_lines.append([cur_object.rectangle.topright, cur_object.rectangle.topleft])
                        corridor_lines.append([cur_object.rectangle.bottomleft, cur_object.rectangle.topleft])
            corridor_lines = np.array(corridor_lines, dtype=np.float32)
            lasers_values = []
            self.lasers_collides = []
            for laser_end_point in self.lasers_end_points:
                # эта функция не работает с коллинеарными
                # есть вариант лучше, но медленней
                # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
                rez = LeaderCorridor_lasers.intersect(corridor_lines[:, 0, :], corridor_lines[:, 1, :],
                                                      np.array([self.host_object.position]),
                                                      np.array([laser_end_point]))
                intersected_line = corridor_lines[rez]
                if len(intersected_line) > 0:
                    x = LeaderCorridor_lasers.seg_intersect(intersected_line[:, 0, :], intersected_line[:, 1, :],
                                                            np.array([self.host_object.position]),
                                                            np.array([laser_end_point]))
                    # TODO: исключить коллинеарные, вместо их точек пересечения добавить ближайшую точку коллинеарной границы
                    # но это бесполезно при использовании функции intersect, которая не работает с коллинеарными
                    exclude_rows = np.concatenate([np.nonzero(np.isinf(x))[0], np.nonzero(np.isnan(x))[0]])
                    norms = np.linalg.norm(x - self.host_object.position, axis=1)
                    lasers_values.append(np.min(norms))
                    closest_dot_idx = np.argmin(np.linalg.norm(x - self.host_object.position, axis=1))
                    self.lasers_collides.append(x[closest_dot_idx])
                else:
                    self.lasers_collides.append(laser_end_point)
        obs = np.ones(self.count_lasers, dtype=np.float32) * self.laser_length
        for i, collide in enumerate(self.lasers_collides):
            obs[i] = np.linalg.norm(collide - self.host_object.position)
        return obs


# Можно конечно через getattr из модуля брать, но так можно проверку добавить
SENSOR_NAME_TO_CLASS = {
    "LaserSensor": LaserSensor,
    "LeaderPositionsTracker": LeaderPositionsTracker,
    "LeaderPositionsTracker_v2": LeaderPositionsTracker_v2,
    "LeaderTrackDetector_vector": LeaderTrackDetector_vector,
    "LeaderTrackDetector_radar": LeaderTrackDetector_radar,
    "LeaderCorridor_lasers": LeaderCorridor_lasers,
    "GreenBoxBorderSensor": GreenBoxBorderSensor,
    "LeaderCorridor_lasers_v2": LeaderCorridor_lasers_v2
}
