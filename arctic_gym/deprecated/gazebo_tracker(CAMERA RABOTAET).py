import tf
import numpy as np

from src.arctic_gym.utils.sensors import LeaderPositionsTracker
from src.arctic_gym.utils.sensors import LeaderTrackDetector_radar
from src.arctic_gym.utils.misc import rotateVector, calculateAngle


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
        delta_cx = delta['delta_x']
        # delta_cx = float(delta_cx)
        delta_cy = delta['delta_y']
        # delta_cy = float(delta_cy)
        # print("XXXXXXXXXXXX_cx_XXXXXXXXXXXX", type(delta_cx))
        # print("XXXXXXXXXXXXX_cy_XXXXXXXXXXX", type(delta_cy))
        follower_position = [0,0]
        # print('!!!!!!!!!!',delta_cx)
        # print('!!!!!!!!!!',delta_cy)
        # print('!!!!!!!!!!',leader_position)
        if len(self.leader_positions_hist)>0:
            # print("LOL", type(self.leader_positions_hist))
            # print("LOL", follower_position)
            x_hist_pose, y_hist_pose = zip(*self.leader_positions_hist)
            # print("XXXXXXXXXXXXXXXXXXXXXXXX", x_hist_pose)
            # print("YYYYYYYYYYYYYYYYYYYYYYYY", y_hist_pose)
            x_new_hist_pose = x_hist_pose - delta_cx
            x_new_hist_pose = np.round(x_new_hist_pose, decimals=2)
            y_new_hist_pose = y_hist_pose - delta_cy
            y_new_hist_pose = np.round(y_new_hist_pose, decimals=2)
            # print("XXXXXXXXXXXXXXXXXXXXXXXX_____", x_new_hist_pose)
            # print("YYYYYYYYYYYYYYYYYYYYYYYY_____", y_new_hist_pose)
            # x_new_hist_pose = x_new_hist_pose.tolist()
            # y_new_hist_pose = y_new_hist_pose.tolist()
            # print("XXXXXXXXXXXXXXXXXXXXXXXX_____", x_new_hist_pose)
            # print("YYYYYYYYYYYYYYYYYYYYYYYY_____", y_new_hist_pose)
            self.leader_positions_hist = list(zip(x_new_hist_pose, y_new_hist_pose))
            # print("DELTA ", delta)
            # print("LEADER POSE  ", leader_position)
            # print("LEADER HIST ", self.leader_positions_hist)
            # print("FOLLOWER ", follower_position)



        self.saving_period = 3
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

    # def scan(self, leader_position, follower_position):
    #     """
    #     Версия сохранения координат для Gazebo, без построения корридора
    #     Args:
    #         leader_position: позиция ведущего робота [x, y]
    #         follower_position: позиция ведомого робота [x, y]
    #     Returns:
    #         История позиций ведущего робота
    #     """
    #
    #     # self.saving_period = 5
    #     if self.saving_counter % self.saving_period == 0:
    #         if len(self.leader_positions_hist) > 0 and (self.leader_positions_hist[-1] == leader_position).all():
    #             return self.leader_positions_hist
    #
    #         self.leader_positions_hist.append(leader_position)
    #
    #     self.saving_counter += 1
    #
    #     """
    #     Удаление точек, на которые наступил робот
    #     footprint робота - круг радиуса 0.9
    #     """
    #     robot_footprint = 1.8
    #     if self.eat_close_points and len(self.leader_positions_hist) > 0:
    #         norms = np.linalg.norm(np.array(self.leader_positions_hist) - follower_position, axis=1)
    #         indexes = np.nonzero(norms <= robot_footprint)[0]
    #         for index in sorted(indexes, reverse=True):
    #             del self.leader_positions_hist[index]
    #
    #     return self.leader_positions_hist


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
        # print("RADAR FOLLOWER POSE", follower_position)
        # print("RADAR FOLLOWER ORIE", direction)
        # print("RADAR LEADER HIST", leader_positions_hist)

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
