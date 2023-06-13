import tf
import rospy
import numpy as np
from scipy.spatial import distance

from src.arctic_gym.utils.sensors import LeaderPositionsTracker
from src.arctic_gym.utils.sensors import LeaderPositionsTracker_v2
from src.arctic_gym.utils.sensors import LeaderTrackDetector_radar
from src.arctic_gym.utils.sensors import LeaderCorridor_lasers
from src.arctic_gym.utils.misc import rotateVector, calculateAngle

from collections import deque

class GazeboLeaderPositionsTracker(LeaderPositionsTracker):
    """
    Gazebo версия отслеживания наблюдаемых позиций лидера.
    """
    def __init__(self, host_object, sensor_name, saving_period):
        super(GazeboLeaderPositionsTracker, self).__init__(host_object,
                                                           sensor_name,
                                                           saving_period=saving_period)

    def scan(self, leader_position, follower_position, delta):
        """
        Версия сохранения координат для Gazebo, без построения корридора
        Args:
            leader_position: позиция ведущего робота [x, y]
            follower_position: позиция ведомого робота [x, y]
        Returns:
            История позиций ведущего робота
        """
        ###
        delta_cx = delta['delta_x']
        delta_cy = delta['delta_y']
        follower_position = [0, 0]

        if len(self.leader_positions_hist) > 0:

            x_hist_pose, y_hist_pose = zip(*self.leader_positions_hist)
            x_new_hist_pose = x_hist_pose - delta_cx
            x_new_hist_pose = np.round(x_new_hist_pose, decimals=2)
            y_new_hist_pose = y_hist_pose - delta_cy
            y_new_hist_pose = np.round(y_new_hist_pose, decimals=2)
            self.leader_positions_hist = list(zip(x_new_hist_pose, y_new_hist_pose))

        # self.saving_period = 3
        ###
        if self.saving_counter % self.saving_period == 0:
            if len(self.leader_positions_hist) > 0 and (self.leader_positions_hist[-1] == leader_position).all():
                return self.leader_positions_hist

            self.leader_positions_hist.append(leader_position)

        self.saving_counter += 1

        """
        Удаление точек, на которые наступил робот
        footprint робота - круг радиуса 0.9
        """
        robot_footprint = 1.8
        if self.eat_close_points and len(self.leader_positions_hist) > 0:
            norms = np.linalg.norm(np.array(self.leader_positions_hist) - follower_position, axis=1)
            indexes = np.nonzero(norms <= robot_footprint)[0]
            for index in sorted(indexes, reverse=True):
                del self.leader_positions_hist[index]

        return self.leader_positions_hist



class GazeboLeaderPositionsTracker_v2(LeaderPositionsTracker_v2):

    def __init__(self, host_object, sensor_name, saving_period):
        super(GazeboLeaderPositionsTracker_v2, self).__init__(host_object,
                                                              sensor_name,
                                                              saving_period=saving_period)

    def scan(self, leader_position, follower_position, delta):
        """
        Версия сохранения координат для Gazebo, без построения корридора
        Args:
            leader_position: позиция ведущего робота [x, y]
            follower_position: позиция ведомого робота [x, y]
        Returns:
            История позиций ведущего робота
        """

        delta_cx = delta['delta_x']
        delta_cy = delta['delta_y']
        follower_position = [0, 0]
        self.generate_corridor = True
        self.max_dev = 2
        # self.max_dev = 35
        leader_max_speed = 1.0 # TODO : потом посмотреть и перенести все в конфиг
        max_distance = 24
        self.saving_period = 3

        print('LEADER', leader_position)
        print('FOLLOWER', follower_position)


        # 1) Пересчет истории
        if len(self.leader_positions_hist) > 0:
            x_hist_pose, y_hist_pose = zip(*self.leader_positions_hist)
            x_new_hist_pose = x_hist_pose - delta_cx
            x_new_hist_pose = np.round(x_new_hist_pose, decimals=5)
            y_new_hist_pose = y_hist_pose - delta_cy
            y_new_hist_pose = np.round(y_new_hist_pose, decimals=5)
            pose_hist = zip(x_new_hist_pose, y_new_hist_pose)
            self.leader_positions_hist.clear()
            self.leader_positions_hist.extend(np.array(x) for x in pose_hist)

        # 2) Пересчет коридора
        if len(self.corridor) > 0:

            right_border, left_border = zip(*self.corridor)

            x_right_border, y_right_border = zip(*right_border)
            x_right_border_new = x_right_border - delta_cx
            y_right_border_new = y_right_border - delta_cy

            r1 = zip(x_right_border_new, y_right_border_new)
            right_border_new = deque()
            right_border_new.extend(np.array(x) for x in r1)

            x_left_border, y_left_border = zip(*left_border)
            x_left_border_new = x_left_border - delta_cx
            # x_left_border_new = np.round(x_left_border_new, decimals=5)
            y_left_border_new = y_left_border - delta_cy
            # y_left_border_new = np.round(y_left_border_new, decimals=5)

            l1 = zip(x_left_border_new, y_left_border_new)
            left_border_new = deque()
            left_border_new.extend(np.array(x) for x in l1)

            self.corridor.clear()
            right_left_new = zip(right_border_new, left_border_new)
            self.corridor.extend(right_left_new)

        if len(self.corridor) > 2 and len(self.leader_positions_hist) > 2 \
                and self.saving_counter % self.saving_period == 0:
            first_point = self.leader_positions_hist[0]
            second_point = self.leader_positions_hist[1]
            while first_point[0] < -0.3 and second_point[0] < -0.1:
            # while first_point[0] < -0.5 and second_point[0] < -0.3 and first_point[1] < -0.5 and second_point[0] < -0.3:
                print("УДАЛИЛИ УДАЛИЛ УДАЛИЛ УДАЛИЛ УДАЛИЛ МЕТОД 1 ")
                self.leader_positions_hist.popleft()
                self.corridor.popleft()
                if len(self.leader_positions_hist) > 2:
                    first_point = self.leader_positions_hist[0]
                    second_point = self.leader_positions_hist[1]
                else:
                    break

        if self.saving_counter % self.saving_period == 0:
        #  3) Проверка на изменение позиции ведущего
            cur_point = leader_position.copy()
            # # Если позиция лидера не изменилась с последнего обсерва, просто возвращаем, что есть, ничего не обновляем
            # if len(self.leader_positions_hist) > 0 and (self.leader_positions_hist[-1] == leader_position).all():
            if len(self.leader_positions_hist) > 0 and np.linalg.norm(cur_point - self.leader_positions_hist[-1]) < 1:
                print('Popal в неизменяемую позицию лидера')
                if self.generate_corridor:
                    return self.leader_positions_hist, self.corridor
                else:
                    return self.leader_positions_hist
        # 4) достраивание точек вначале симуляции плюс добавление в историю
            if len(self.leader_positions_hist) == 0 and self.saving_counter == 0:
                # first_dots_for_follower_count = int(distance.euclidean(follower_position, leader_position) /
                #                                     (self.saving_period * 1.5 * leader_max_speed))
                first_dots_for_follower_count = 5
                self.leader_positions_hist.extend(np.array(x) for x in
                                                    zip(np.linspace(follower_position[0], leader_position[0],
                                                                    first_dots_for_follower_count),
                                                        np.linspace(follower_position[1], leader_position[1],
                                                                    first_dots_for_follower_count)))
            else:
                last_point = self.leader_positions_hist[-1]
                last_dist = np.linalg.norm(last_point - follower_position)
                current_point = leader_position.copy()
                new_dist = np.linalg.norm(current_point - follower_position)
                print("РАЗНИЦА МЕЖДУ ДВУМЯ ТОЧКАМИ НОВОЙ И ПОСЛЕДНЕЙ", new_dist)
                if new_dist > last_dist and new_dist < 25:
                    self.leader_positions_hist.append(leader_position.copy())

        # TODO : 5) Удаление точек
            # удаление точек коридора и истории
            # TODO : удаление точек истории (альтернативный метод ...)
            dists = np.linalg.norm(np.array(self.leader_positions_hist)[:-1, :] -
                                   np.array(self.leader_positions_hist)[1:, :], axis=1)
            path_length = np.sum(dists)
            while path_length > max_distance:
                print("УДАЛИЛИ УДАЛИЛ УДАЛИЛ УДАЛИЛ УДАЛИЛ МЕТОД 2")
                if len(self.leader_positions_hist) > 0:
                    self.leader_positions_hist.popleft()
                if len(self.corridor) > 0:
                    self.corridor.popleft()
                dists = np.linalg.norm(np.array(self.leader_positions_hist)[:-1, :] -
                                       np.array(self.leader_positions_hist)[1:, :], axis=1)
                path_length = np.sum(dists)

            # 6) Генерация коридора
            if self.generate_corridor and len(self.leader_positions_hist) > 1:
                if self.saving_counter == 0:
                    for i in range(len(self.leader_positions_hist) - 1, 0, -1):
                        last_2points_vec = self.leader_positions_hist[i] - self.leader_positions_hist[i - 1]
                        last_2points_vec *= self.max_dev / np.linalg.norm(last_2points_vec)
                        right_border_dot = rotateVector(last_2points_vec, 90)
                        right_border_dot += self.leader_positions_hist[-i - 1]
                        left_border_dot = rotateVector(last_2points_vec, -90)
                        left_border_dot += self.leader_positions_hist[-i - 1]
                        self.corridor.append([right_border_dot, left_border_dot])
                last_2points_vec = np.array(self.leader_positions_hist[-1]) - np.array(self.leader_positions_hist[-2])
                last_2points_vec *= self.max_dev / np.linalg.norm(last_2points_vec)
                right_border_dot = rotateVector(last_2points_vec, 90)
                right_border_dot += self.leader_positions_hist[-2]
                left_border_dot = rotateVector(last_2points_vec, -90)
                left_border_dot += self.leader_positions_hist[-2]
                self.corridor.append([right_border_dot, left_border_dot])

        self.saving_counter += 1
        print("ИСТОРИЯ И КОРИДОР", self.leader_positions_hist)
        return self.leader_positions_hist, self.corridor


class GazeboLeaderPositionsTrackerRadar(LeaderTrackDetector_radar):

    def __init__(self, max_distance, host_object, sensor_name, position_sequence_length, detectable_positions, radar_sectors_number):
        super(GazeboLeaderPositionsTrackerRadar, self).__init__(host_object,
                                                                sensor_name,
                                                                position_sequence_length,
                                                                detectable_positions,
                                                                radar_sectors_number)

        self.max_distance = max_distance

    def scan(self, follower_position, follower_orientation, leader_positions_hist):
        """
        direction - угол от 0 до 360, где 0 - робот смотрит направо
        """
        _, _, yaw = tf.transformations.euler_from_quaternion(follower_orientation)
        direction = np.degrees(yaw)
        # direction = np.degrees(follower_orientation)
        follower_position = [0, 0]

        follower_dir_vec = rotateVector(np.array([1, 0]), direction)
        follower_right_dir = direction + 90
        if follower_right_dir >= 360:
            follower_right_dir -= 360
        follower_right_vec = rotateVector(np.array([1, 0]), follower_right_dir)

        self.radar_values = np.zeros(self.radar_sectors_number, dtype=np.float32)

        if len(leader_positions_hist) > 0:

            if self.detectable_positions == 'near':
                leader_positions_hist = np.array(leader_positions_hist)
                vecs_follower_to_leadhistory = leader_positions_hist - follower_position
                distances_follower_to_chosenDots = np.linalg.norm(vecs_follower_to_leadhistory, axis=1)
                closest_indexes = np.argsort(distances_follower_to_chosenDots)
                vecs_follower_to_leadhistory = vecs_follower_to_leadhistory[closest_indexes]
                distances_follower_to_chosenDots = distances_follower_to_chosenDots[closest_indexes]
            else:
                if self.detectable_positions == "new":
                    chosen_dots = np.array(leader_positions_hist[-self.position_sequence_length:])
                elif self.detectable_positions == "old":
                    chosen_dots = np.array(leader_positions_hist[:self.position_sequence_length])
                vecs_follower_to_leadhistory = chosen_dots - follower_position
                distances_follower_to_chosenDots = np.linalg.norm(vecs_follower_to_leadhistory, axis=1)

            angles_history_to_dir = calculateAngle(vecs_follower_to_leadhistory, follower_dir_vec)
            angles_history_to_right = calculateAngle(vecs_follower_to_leadhistory, follower_right_vec)
            angles_history_to_right[angles_history_to_dir > np.pi / 2] = -angles_history_to_right[angles_history_to_dir > np.pi / 2]

            for i in range(self.radar_sectors_number):
                sector_dots_distances = distances_follower_to_chosenDots[
                    (angles_history_to_right >= self.sectorsAngle_rad * i) & (
                            angles_history_to_right < self.sectorsAngle_rad * (i + 1))]
                if len(sector_dots_distances) > 0:
                    self.radar_values[i] = np.min(sector_dots_distances)

        self.radar_values = np.clip(self.radar_values / self.max_distance, 0, 1)

        return self.radar_values[::-1]

class GazeboLeaderPositionsCorridorLasers(LeaderCorridor_lasers):
    def __init__(self, max_distance, host_object, sensor_name, position_sequence_length, react_to_safe_corridor, react_to_obstacles,
                 react_to_green_zone, front_lasers_count, back_lasers_count):
        super(GazeboLeaderPositionsCorridorLasers, self).__init__(host_object, sensor_name, react_to_safe_corridor, react_to_obstacles,
                 react_to_green_zone, front_lasers_count, back_lasers_count)

        self.front_lasers_count = front_lasers_count
        self.back_lasers_count = back_lasers_count

    def scan(self, follower_position, follower_orientation, leader_positions_hist, corridor,
             cur_object_points_1, cur_object_points_2):

        # print(cur_object_points)
        self.lasers_collides = []
        self.lasers_end_points = []
        self.laser_length = 4
        follower_position = [0, 0]

        self.react_to_safe_corridor = True
        self.react_to_green_zone = True
        self.react_to_obstacles = True

        self.front_lasers_count = 5
        self.back_lasers_count = 2

        # TODO : проверить ориентацию

        _, _, yaw = tf.transformations.euler_from_quaternion(follower_orientation)
        direction = np.degrees(yaw)
        print("DIRECTION", direction)

        follower_orientation = direction


        self.lasers_end_points.append(
            follower_position + rotateVector(np.array([self.laser_length, 0]), follower_orientation + 40))
        self.lasers_end_points.append(
            follower_position + rotateVector(np.array([self.laser_length, 0]), follower_orientation))
        self.lasers_end_points.append(
            follower_position + rotateVector(np.array([self.laser_length, 0]), follower_orientation - 40))

        if self.front_lasers_count == 5:
            self.lasers_end_points.append(
                follower_position + rotateVector(np.array([self.laser_length, 0]),
                                                 follower_orientation + 90))
            self.lasers_end_points.append(
                follower_position + rotateVector(np.array([self.laser_length, 0]),
                                                 follower_orientation - 90))
        if self.back_lasers_count == 2:
            self.lasers_end_points.append(
                follower_position + rotateVector(np.array([self.laser_length, 0]),
                                                 follower_orientation + 150))
            self.lasers_end_points.append(
                follower_position + rotateVector(np.array([self.laser_length, 0]),
                                                 follower_orientation - 150))

        if len(corridor) > 1:
            corridor_lines = list()
            if self.react_to_safe_corridor:
                for i in range(len(corridor) - 1):
                    corridor_lines.append([corridor[i][0], corridor[i + 1][0]])
                    corridor_lines.append([corridor[i][1], corridor[i + 1][1]])
            if self.react_to_green_zone:
                corridor_lines.append([corridor[0][0], corridor[0][1]])
                # TODO : убрал из изначального варианта для предотвращения ошибок
                # corridor_lines.append([corridor[-1][0], corridor[-1][1]])

            if self.react_to_obstacles and len(cur_object_points_1) > 1:

                cur_object_points_1 = np.array(cur_object_points_1)
                cur_object_points_2 = np.array(cur_object_points_2)

                for i in range(len(cur_object_points_1) - 1):

                    if np.linalg.norm(cur_object_points_1[i] - cur_object_points_1[i+1]) < 0.5:
                        corridor_lines.append([cur_object_points_1[i], cur_object_points_1[i+1]])
                    else:
                        corridor_lines.append([cur_object_points_1[i], cur_object_points_2[i]])

            corridor_lines = np.array(corridor_lines, dtype=np.float32)
            lasers_values = []
            self.lasers_collides = []

            for laser_end_point in self.lasers_end_points:
                # эта функция не работает с коллинеарными
                # есть вариант лучше, но медленней
                # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
                # print(len(corridor_lines[:, 0, :]))

                rez = LeaderCorridor_lasers.intersect(corridor_lines[:, 0, :], corridor_lines[:, 1, :],
                                                      np.array([follower_position]),
                                                      np.array([laser_end_point]))
                intersected_line = corridor_lines[rez]

                if len(intersected_line) > 0:
                    x = LeaderCorridor_lasers.seg_intersect(intersected_line[:, 0, :], intersected_line[:, 1, :],
                                                            np.array([follower_position]),
                                                            np.array([laser_end_point]))
                    # TODO: исключить коллинеарные, вместо их точек пересечения добавить ближайшую точку коллинеарной границы
                    # но это бесполезно при использовании функции intersect, которая не работает с коллинеарными
                    exclude_rows = np.concatenate([np.nonzero(np.isinf(x))[0], np.nonzero(np.isnan(x))[0]])
                    norms = np.linalg.norm(x - follower_position, axis=1)
                    lasers_values.append(np.min(norms))
                    closest_dot_idx = np.argmin(np.linalg.norm(x - follower_position, axis=1))
                    self.lasers_collides.append(x[closest_dot_idx])
                else:
                    self.lasers_collides.append(laser_end_point)
        obs = np.ones(self.front_lasers_count + self.back_lasers_count, dtype=np.float32) * self.laser_length

        for i, collide in enumerate(self.lasers_collides):
            obs[i] = np.linalg.norm(collide - follower_position)

        self.laser_values_obs = obs
        self.laser_values_obs = (self.laser_values_obs / self.laser_length)
        self.laser_values_obs[5] = self.laser_values_obs[5]*0.65
        self.laser_values_obs[6] = self.laser_values_obs[6]*0.65

        print(' ')
        print(self.laser_values_obs)
        return self.laser_values_obs


