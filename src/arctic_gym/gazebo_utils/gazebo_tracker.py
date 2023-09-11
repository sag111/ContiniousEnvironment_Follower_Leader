import numpy as np

from math import cos, sin, radians

from src.continuous_grid_arctic.utils.misc import rotateVector, angle_correction
from src.continuous_grid_arctic.utils.sensors import LeaderPositionsTracker_v2
from src.continuous_grid_arctic.utils.sensors import LeaderCorridor_lasers
from src.continuous_grid_arctic.utils.sensors import LeaderCorridor_Prev_lasers_v2

from collections import deque


class GazeboLeaderPositionsTracker_v2(LeaderPositionsTracker_v2):

    def __init__(self, *args, **kwargs):
        super(GazeboLeaderPositionsTracker_v2, self).__init__(*args, **kwargs)

    def scan(self, leader_position, follower_position, follower_orientation, delta):
        """
        Версия сохранения координат для Gazebo, без построения корридора
        Args:
            leader_position: позиция ведущего робота [x, y]
            follower_position: позиция ведомого робота [x, y]
        Returns:
            История позиций ведущего робота
        """

        follower_orientation = np.degrees(follower_orientation)[2]

        delta_cx = delta['delta_x']
        delta_cy = delta['delta_y']
        follower_position = [0, 0]
        self.generate_corridor = True
        # ширина коридора
        self.max_dev = 2
        # self.max_dev = 35
        leader_max_speed = 1.0 # TODO : потом посмотреть и перенести все в конфиг

        # длина коридора
        max_distance = 25
        self.saving_period = 3

        # print('LEADER', leader_position)
        # print('FOLLOWER', follower_position)

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

        # Баг с удалением коридора
        # if len(self.corridor) > 2 and len(self.leader_positions_hist) > 2 \
        #         and self.saving_counter % self.saving_period == 0:
        #     first_point = self.leader_positions_hist[0]
        #     second_point = self.leader_positions_hist[1]
        #     while first_point[0] < -0.3 and second_point[0] < -0.1:
        #     # while first_point[0] < -0.5 and second_point[0] < -0.3 and first_point[1] < -0.5 and second_point[0] < -0.3:
        #     #     print("УДАЛИЛИ УДАЛИЛ УДАЛИЛ УДАЛИЛ УДАЛИЛ МЕТОД 1 ")
        #         self.leader_positions_hist.popleft()
        #         self.corridor.popleft()
        #         if len(self.leader_positions_hist) > 2:
        #             first_point = self.leader_positions_hist[0]
        #             second_point = self.leader_positions_hist[1]
        #         else:
        #             break

        if self.saving_counter % self.saving_period == 0:
        #  3) Проверка на изменение позиции ведущего
            cur_point = leader_position.copy()
            # # Если позиция лидера не изменилась с последнего обсерва, просто возвращаем, что есть, ничего не обновляем
            # if len(self.leader_positions_hist) > 0 and (self.leader_positions_hist[-1] == leader_position).all():
            if len(self.leader_positions_hist) > 0 and np.linalg.norm(cur_point - self.leader_positions_hist[-1]) < 1:
                # print('Popal в неизменяемую позицию лидера')
                if self.generate_corridor:
                    return self.leader_positions_hist, self.corridor
                else:
                    return self.leader_positions_hist
        # 4) достраивание точек вначале симуляции плюс добавление в историю
            if len(self.leader_positions_hist) == 0 and self.saving_counter == 0:
                point_start_distance_behind_follower = 10
                point_start_position_theta = angle_correction(follower_orientation + 180)
                point_behind_follower = np.array(
                    (point_start_distance_behind_follower * cos(radians(point_start_position_theta)),
                     point_start_distance_behind_follower * sin(radians(point_start_position_theta)))) \
                                        + follower_position

                first_dots_for_follower_count = 10
                # first_dots_for_follower_count = int(
                #     distance.euclidean(point_behind_follower, env.leader.position) / (
                #             self.saving_period * 5 * env.leader.max_speed))

                self.leader_positions_hist.extend(np.array(x) for x in
                                                  zip(np.linspace(point_behind_follower[0], leader_position[0],
                                                                  first_dots_for_follower_count),
                                                      np.linspace(point_behind_follower[1], leader_position[1],
                                                                  first_dots_for_follower_count)))
                # first_dots_for_follower_count = int(distance.euclidean(follower_position, leader_position) /
                #                                     (self.saving_period * 1.5 * leader_max_speed))
                # first_dots_for_follower_count = 5
                # self.leader_positions_hist.extend(np.array(x) for x in
                #                                     zip(np.linspace(follower_position[0], leader_position[0],
                #                                                     first_dots_for_follower_count),
                #                                         np.linspace(follower_position[1], leader_position[1],
                #                                                     first_dots_for_follower_count)))
            else:
                last_point = self.leader_positions_hist[-1]
                last_dist = np.linalg.norm(last_point - follower_position)
                current_point = leader_position.copy()
                new_dist = np.linalg.norm(current_point - follower_position)
                # print("РАЗНИЦА МЕЖДУ ДВУМЯ ТОЧКАМИ НОВОЙ И ПОСЛЕДНЕЙ", new_dist)
                if new_dist > last_dist and new_dist < 25:
                    self.leader_positions_hist.append(leader_position.copy())

            # удаление точек коридора и истории
            # TODO : удаление точек истории (альтернативный метод ...)
            dists = np.linalg.norm(np.array(self.leader_positions_hist)[:-1, :] -
                                   np.array(self.leader_positions_hist)[1:, :], axis=1)
            path_length = np.sum(dists)

            while path_length > max_distance:
                # print("УДАЛИЛИ УДАЛИЛ УДАЛИЛ УДАЛИЛ УДАЛИЛ МЕТОД 2")
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
        # print("ИСТОРИЯ И КОРИДОР", self.leader_positions_hist)

        return self.leader_positions_hist, self.corridor


class GazeboCorridor_Prev_lasers_v2(LeaderCorridor_Prev_lasers_v2):

    def __init__(self, *args, **kwargs):
        super(GazeboCorridor_Prev_lasers_v2, self).__init__(*args, **kwargs)

    def collect_obstacle_edges(self, corridor, cur_object_points_1, cur_object_points_2):
        obstacle_lines = list()
        if self.react_to_safe_corridor:
            for i in range(len(corridor) - 1):
                obstacle_lines.append([corridor[i][0], corridor[i + 1][0]])
                obstacle_lines.append([corridor[i][1], corridor[i + 1][1]])
        if self.react_to_green_zone:
            obstacle_lines.append([corridor[0][0], corridor[0][1]])
            obstacle_lines.append([corridor[-1][0], corridor[-1][1]])
        if self.react_to_obstacles:
            cur_object_points_1 = np.array(cur_object_points_1)
            cur_object_points_2 = np.array(cur_object_points_2)

            for i in range(len(cur_object_points_1) - 1):
                if np.linalg.norm(cur_object_points_1[i] - cur_object_points_1[i + 1]) < 0.5:
                    obstacle_lines.append([cur_object_points_1[i], cur_object_points_1[i + 1]])
                else:
                    obstacle_lines.append([cur_object_points_1[i], cur_object_points_2[i]])

        obstacle_lines = np.array(obstacle_lines, dtype=np.float32)

        return obstacle_lines

    def scan(self, follower_position, follower_orientation, corridor, cur_object_points_1, cur_object_points_2):
        self.lasers_collides = []
        self.lasers_end_points = []

        follower_orientation = np.degrees(follower_orientation)[2]

        for i in range(self.lasers_count):
            self.lasers_end_points.append(follower_position + rotateVector(np.array([self.laser_length, 0]),
                                                            (follower_orientation-45) + i*self.laser_period))

        if len(corridor) > 1:
            corridor_lines = self.collect_obstacle_edges(corridor, cur_object_points_1, cur_object_points_2)
            self.history_obstacles_list.pop(0)
            self.history_obstacles_list.append(corridor_lines)

            all_obs_list = []

            for j, corridor_lines_item in enumerate(self.history_obstacles_list):
                corridor_lines_item = np.array(corridor_lines_item)

                lasers_values_item = []
                self.lasers_collides_item = []
                for laser_end_point in self.lasers_end_points:
                    if corridor_lines_item != []:
                        rez = LeaderCorridor_lasers.intersect(corridor_lines_item[:, 0, :],
                                                              corridor_lines_item[:, 1, :],
                                                              np.array([follower_position]),
                                                              np.array([laser_end_point]))
                        intersected_line_item = corridor_lines_item[rez]

                        if len(intersected_line_item) > 0:
                            x = LeaderCorridor_lasers.seg_intersect(intersected_line_item[:, 0, :],
                                                                    intersected_line_item[:, 1, :],
                                                                    np.array([follower_position]),
                                                                    np.array([laser_end_point]))
                            # TODO: исключить коллинеарные, вместо их точек пересечения добавить ближайшую точку коллинеарной границы
                            # но это бесполезно при использовании функции intersect, которая не работает с коллинеарными
                            exclude_rows = np.concatenate([np.nonzero(np.isinf(x))[0], np.nonzero(np.isnan(x))[0]])
                            norms = np.linalg.norm(x - follower_position, axis=1)
                            lasers_values_item.append(np.min(norms))
                            closest_dot_idx = np.argmin(np.linalg.norm(x - follower_position, axis=1))
                            self.lasers_collides_item.append(x[closest_dot_idx])
                        else:
                            self.lasers_collides_item.append(laser_end_point)
                    else:
                        self.lasers_collides_item.append(laser_end_point)

                obs_item = np.ones(self.lasers_count, dtype=np.float32) * self.laser_length
                for i, collide in enumerate(self.lasers_collides_item):
                    obs_item[i] = np.linalg.norm(collide - follower_position)

                if self.pad_sectors:
                    front = np.zeros(len(obs_item))
                    right = np.zeros(len(obs_item))
                    behind = np.zeros(len(obs_item))
                    left = np.zeros(len(obs_item))

                    lasers_in_sector = self.lasers_count / 4
                    for i in range(len(obs_item)):
                        if i < lasers_in_sector:
                            front[i] = obs_item[i]
                        elif lasers_in_sector <= i < 2 * lasers_in_sector:
                            right[i] = obs_item[i]
                        elif 2 * lasers_in_sector <= i < 3 * lasers_in_sector:
                            behind[i] = obs_item[i]
                        else:
                            left[i] = obs_item[i]

                    # front = np.array([obs_item[0], obs_item[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, obs_item[11]])
                    # right = np.array([0, 0, obs_item[2], obs_item[3], obs_item[4], 0, 0, 0, 0, 0, 0, 0])
                    # behind = np.array([0, 0, 0, 0, 0, obs_item[5], obs_item[6], obs_item[7], 0, 0, 0, 0])
                    # left = np.array([0, 0, 0, 0, 0, 0, 0, 0, obs_item[8], obs_item[9], obs_item[10], 0])
                    res_out = np.concatenate((front, right, behind, left), axis=None)
                else:
                    res_out = obs_item

                all_obs_list.append(res_out)

            all_obs_arr = np.array(all_obs_list)
            # print(all_obs_arr)
        #             print('ALL CORIDOR OBS ARR 1: ', all_obs_arr)
        #             print('ALL CORIDOR OBS ARR 1: ', all_obs_arr.shape)
        return all_obs_arr
