import gym
from collections import deque
import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box
try:
    from utils.misc import rotateVector, calculateAngle
except:
    from continuous_grid_arctic.utils.misc import rotateVector, calculateAngle

from warnings import warn


class MyFrameStack(ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    .. note::
        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
    .. note::
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.
    Example::
        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally
    """

    def __init__(self, env, framestack, lz4_compress=False):
        super().__init__(env)
        self.framestack = framestack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=framestack)

        low = np.tile(self.observation_space.low[...], framestack)
        high = np.tile(
            self.observation_space.high[...], framestack
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self):
        assert len(self.frames) == self.framestack, (len(self.frames), self.framestack)
        observes = np.concatenate(self.frames)

        return observes
        # return gym.wrappers.frame_stack.LazyFrames(observes, self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.framestack)]
        return self.observation()


class ContinuousObserveModifier_v0(ObservationWrapper):

    def __init__(self, env, action_values_range=None, lz4_compress=False):
        super().__init__(env)
        self.observations_list = None
        features_number = 0
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", env)
        print(env.env.use_prev_obs)
        self.prev_obs_flag = env.env.use_prev_obs
        self.num_prev_obs = env.env.max_prev_obs

        # этот должен быть -1:1
        if 'LeaderTrackDetector_vector' in self.follower_sensors:
            features_number += env.follower_sensors['LeaderTrackDetector_vector']['position_sequence_length'] * 2
        # этот должен быть 0:1
        if 'LeaderTrackDetector_radar' in self.follower_sensors:
            features_number += env.follower_sensors['LeaderTrackDetector_radar']['radar_sectors_number']
        # этот должен быть 0:1
        if 'LeaderCorridor_lasers' in self.follower_sensors:
            if 'front_lasers_count' in env.follower_sensors['LeaderCorridor_lasers']:
                features_number += env.follower_sensors['LeaderCorridor_lasers']['front_lasers_count']
            else:
                features_number += 3  # env.follower_sensors['LeaderCorridor_lasers']['lasers_count']
            if 'back_lasers_count' in env.follower_sensors['LeaderCorridor_lasers']:
                features_number += env.follower_sensors['LeaderCorridor_lasers']['back_lasers_count']

        # TODO : привести потом в нормальный вид
        if 'LeaderCorridor_lasers_v2' in self.follower_sensors:
            if 'front_lasers_count' in env.follower_sensors['LeaderCorridor_lasers_v2']:
                features_number += env.follower_sensors['LeaderCorridor_lasers_v2']['front_lasers_count']
            if 'back_lasers_count' in env.follower_sensors['LeaderCorridor_lasers_v2']:
                features_number += env.follower_sensors['LeaderCorridor_lasers_v2']['back_lasers_count']

        # TODO : привести потом в нормальный вид
        if 'LeaderObstacles_lasers' in self.follower_sensors:
            if 'front_lasers_count' in env.follower_sensors['LeaderObstacles_lasers']:
                features_number += env.follower_sensors['LeaderObstacles_lasers']['front_lasers_count']
            if 'back_lasers_count' in env.follower_sensors['LeaderObstacles_lasers']:
                features_number += env.follower_sensors['LeaderObstacles_lasers']['back_lasers_count']

        # TODO : привести потом в нормальный вид
        if 'Leader_Dyn_Obstacles_lasers' in self.follower_sensors:
            if 'front_lasers_count' in env.follower_sensors['Leader_Dyn_Obstacles_lasers']:
                features_number += env.follower_sensors['Leader_Dyn_Obstacles_lasers']['front_lasers_count']
            if 'back_lasers_count' in env.follower_sensors['Leader_Dyn_Obstacles_lasers']:
                features_number += env.follower_sensors['Leader_Dyn_Obstacles_lasers']['back_lasers_count']

        if 'FollowerInfo' in self.follower_sensors:
            if 'speed_direction_param' in env.follower_sensors['FollowerInfo']:
                features_number += env.follower_sensors['FollowerInfo']['speed_direction_param']

        if 'LaserSensor' in self.follower_sensors:
            if self.follower_sensors['LaserSensor']['return_all_points']:
                lidar_points_number = (int(self.follower_sensors['LaserSensor']['available_angle'] / self.follower_sensors['LaserSensor']['angle_step'])+1) * self.follower_sensors['LaserSensor']['points_number']
            else:
                lidar_points_number = (int(self.follower_sensors['LaserSensor']['available_angle'] / self.follower_sensors['LaserSensor']['angle_step'])+1)
            if self.follower_sensors['LaserSensor']["return_only_distances"]:
                features_number += lidar_points_number
            else:
                features_number += lidar_points_number * 2

        self.features_number_num = features_number
        if self.prev_obs_flag:
            self.observation_space = Box(-np.ones([self.num_prev_obs, features_number]),
                                         np.ones([self.num_prev_obs, features_number]))
        else:
            self.observation_space = Box(-np.ones(features_number),
                                         np.ones(features_number))

        self.action_values_range = action_values_range
        if self.action_values_range is not None:
            low_bound, high_bound = self.action_values_range
            self.scale = (high_bound - low_bound) / (env.action_space.high - env.action_space.low)
            self.min = low_bound - env.action_space.low * self.scale
            self.action_space = Box(low=-np.ones_like(env.action_space.low),
                                    high=np.ones_like(env.action_space.high),
                                    shape=env.action_space.shape,
                                    dtype=env.action_space.dtype)

    def observation(self, obs):
        features_list = []
        if 'LeaderTrackDetector_vector' in self.follower.sensors:
            history_vecs = obs['LeaderTrackDetector_vector'].flatten()
            history_vecs = np.clip(history_vecs / self.max_distance, -1, 1)
            features_list.append(history_vecs)
        if 'LeaderTrackDetector_radar' in self.follower.sensors:
            history_radar = obs['LeaderTrackDetector_radar']
            history_radar = np.clip(history_radar / self.max_distance, 0, 1)
            features_list.append(history_radar)
        if 'LeaderCorridor_lasers' in self.follower.sensors:
            corridor_lasers = obs['LeaderCorridor_lasers']
            corridor_lasers =  np.clip(corridor_lasers / self.follower.sensors['LeaderCorridor_lasers'].laser_length, 0, 1)
            features_list.append(corridor_lasers)

        if 'LeaderCorridor_lasers_v2' in self.follower.sensors:
            corridor_lasers_v2 = obs['LeaderCorridor_lasers_v2']
            corridor_lasers_v2 = np.clip(corridor_lasers_v2 / self.follower.sensors['LeaderCorridor_lasers_v2'].laser_length, 0, 1)
            features_list.append(corridor_lasers_v2)

        if 'LeaderObstacles_lasers' in self.follower.sensors:
            corridor_obs_lasers = obs['LeaderObstacles_lasers']
            corridor_obs_lasers = np.clip(corridor_obs_lasers / self.follower.sensors['LeaderObstacles_lasers'].laser_length, 0, 1)
            features_list.append(corridor_obs_lasers)

        if 'Leader_Dyn_Obstacles_lasers' in self.follower.sensors:
            corridor_obs_lasers = obs['Leader_Dyn_Obstacles_lasers']
            corridor_obs_lasers = np.clip(corridor_obs_lasers / self.follower.sensors['Leader_Dyn_Obstacles_lasers'].laser_length, 0, 1)
            features_list.append(corridor_obs_lasers)

        if 'FollowerInfo' in self.follower.sensors:
            follower_info = obs['FollowerInfo']
            follower_info = np.clip(follower_info, -1, 1)
            features_list.append(follower_info)

        if 'LaserSensor' in self.follower_sensors:
            lidar_sensed_points = obs['LaserSensor']
            # переход к относительным координатам лучше делать в сенсоре
            if len(lidar_sensed_points.shape)==2:
                lidar_sensed_points = np.concatenate(lidar_sensed_points)
            lidar_sensed_points = np.clip(lidar_sensed_points / (self.follower.sensors["LaserSensor"].range * self.PIXELS_TO_METER), -1, 1)
            features_list.append(lidar_sensed_points)

        if self.prev_obs_flag:
            concatenate_features_list = np.concatenate(features_list)
            self.observations_list = self.add_prev_obs(concatenate_features_list)
        else:
            self.observations_list = np.concatenate(features_list)
        print("observations_list", self.observations_list)
        return self.observations_list

    def add_prev_obs(self, concatenate_features_list):
        if self.observations_list is None:
            self.observations_list = np.zeros([self.num_prev_obs, self.features_number_num])

        remove_arr = self.observations_list
        # вариант добавления нового сверху
        # after_remove = np.delete(remove_arr, [-1], 0)
        # after_add = np.insert(after_remove, 0, concatenate_features_list, axis=0)
        # вариант добавления нового снизу

        after_remove = np.delete(remove_arr, [0], 0)
        after_add = np.vstack([after_remove, after_remove])
        return after_add

    def step(self, action):
        if self.action_values_range is not None:
            action -= self.min
            action /= self.scale
        obs, rews, dones, infos = self.env.step(action)
        obs = self.observation(obs)
        print("OBS",obs)
        return obs, rews, dones, infos


class ContinuousObserveModifierPrev(ObservationWrapper):

    """

    Враппер для накопления предыдущих значений двух модернизированных сенсоров (1) Лучевой сенсор с 12 лучами на коридор
    и препятствия; 2) Лучевой сенсор на препятствия с 30 (вариативно) лучами

    """

    def __init__(self, env, action_values_range=None, lz4_compress=False):
        super().__init__(env)
        self.observations_list = None
        features_number = 0
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", env)
        print(env.env.use_prev_obs)
        self.prev_obs_flag = env.env.use_prev_obs
        self.num_prev_obs = env.env.max_prev_obs


        if 'LeaderCorridor_Prev_lasers_v2' in self.follower_sensors:
            if 'front_lasers_count' in env.follower_sensors['LeaderCorridor_Prev_lasers_v2']:
                features_number += env.follower_sensors['LeaderCorridor_Prev_lasers_v2']['front_lasers_count']
            if 'back_lasers_count' in env.follower_sensors['LeaderCorridor_Prev_lasers_v2']:
                features_number += env.follower_sensors['LeaderCorridor_Prev_lasers_v2']['back_lasers_count']


        if 'LaserPrevSensor' in self.follower_sensors:
            if 'front_lasers_count' in env.follower_sensors['LaserPrevSensor']:
                features_number += env.follower_sensors['LaserPrevSensor']['front_lasers_count']
            if 'back_lasers_count' in env.follower_sensors['LaserPrevSensor']:
                features_number += env.follower_sensors['LaserPrevSensor']['back_lasers_count']


        self.features_number_num = features_number

        self.observation_space = Box(-np.ones([self.num_prev_obs, features_number]),
                                     np.ones([self.num_prev_obs, features_number]))


        self.action_values_range = action_values_range
        if self.action_values_range is not None:
            low_bound, high_bound = self.action_values_range
            self.scale = (high_bound - low_bound) / (env.action_space.high - env.action_space.low)
            self.min = low_bound - env.action_space.low * self.scale
            self.action_space = Box(low=-np.ones_like(env.action_space.low),
                                    high=np.ones_like(env.action_space.high),
                                    shape=env.action_space.shape,
                                    dtype=env.action_space.dtype)

    # TODO: исправить
    def observation(self, obs):
        features_list = []

        # TODO: исправить
        if 'LeaderCorridor_Prev_lasers_v2' in self.follower.sensors:
            corridor_lasers_v2 = obs['LeaderCorridor_Prev_lasers_v2']
            corridor_lasers_v2 = np.clip(corridor_lasers_v2 / self.follower.sensors['LeaderCorridor_Prev_lasers_v2'].laser_length, 0, 1)
            # features_list.append(corridor_lasers_v2)

        if 'LaserPrevSensor' in self.follower.sensors:
            corridor_obs_lasers = obs['LaserPrevSensor']
            corridor_obs_lasers = np.clip(corridor_obs_lasers / self.follower.sensors['LaserPrevSensor'].laser_length, 0, 1)
            # features_list.append(corridor_obs_lasers)


        self.observations_list  = np.concatenate((corridor_lasers_v2, corridor_obs_lasers), axis=1)

        print("observations_list", self.observations_list)
        return self.observations_list

    def step(self, action):
        if self.action_values_range is not None:
            action -= self.min
            action /= self.scale
        obs, rews, dones, infos = self.env.step(action)
        obs = self.observation(obs)
        print("OBS",obs)
        return obs, rews, dones, infos

# сначала сделал нормализацию в отдельном классе, а не параметром action_values_range
# просто для обратной совместимости оставил, чтоб старые конфиги работали.
class ContinuousObserveModifier_v1(ContinuousObserveModifier_v0):
    def __init__(self, env, lz4_compress=False):
        super().__init__(env, action_values_range=[-1, 1])
        warn("ContinuousObserveModifier_v1 is deprecated and will be removed", DeprecationWarning)

class LeaderTrajectory_v0(ObservationWrapper):
    """
    Устаревший класс, нужен только для проверки обратной совместимости с экспериемнтами, запущенными на коммите 86211bf4a3b0406e23bc561c00e1ea975c20f90b
    """

    def __init__(self, env, framestack, radar_sectors_number, lz4_compress=False):
        super().__init__(env)
        self.framestack = framestack
        self.leader_positions_hist = list()
        self.radar_sectors_number = radar_sectors_number
        self.sectorsAngle = np.pi / radar_sectors_number

        self.observation_space = Box(-np.ones(self.framestack * 2 + radar_sectors_number),
                                     np.ones(self.framestack * 2 + radar_sectors_number))

    def observation(self, obs):
        """
        На вход ожидается вектор с 13 компонентами:
        - позиция х лидера
        - позиция y лидера
        - скорость лидера
        - направление лидера
        - скорость поворота лидера
        - позиция х фолловера
        - позиция y фолловера
        - скорость фолловера
        - направление фолловера
        - скорость поворота фолловера
        - минимальная дистанция
        - максимальная дистанция
        - максимальное отклонение от маршрута

        На выходе вектор из 2 конкатеринованных вектора:
        - вектор из пар координат векторов разностей между текущей позицией ведомого и последними N позициями ведущего по которым он прошёл
        - вектор радара, имеет количество компонент равное аргументу из конфига radar_sectors_number, каждая компонента дистанция до ближайшей точки в соответствующем секторе полугруга перед собой.
        """
        # change leader absolute pos, speed, direction to relative
        # self.leader_positions_hist.append(obs[:2])
        vecs_follower_to_leadhistory_far = np.zeros((self.framestack, 2), dtype=np.float32)
        if len(self.leader_positions_hist) > 0:
            vecs = np.array(self.leader_positions_hist[-self.framestack:]) - obs['numerical_features'][5:7]
            vecs_follower_to_leadhistory_far[:min(len(self.leader_positions_hist), self.framestack)] = vecs
        vecs_follower_to_leadhistory_far = vecs_follower_to_leadhistory_far.flatten()
        vecs_follower_to_leadhistory_far = np.clip(vecs_follower_to_leadhistory_far / (self.max_distance * 2), -1, 1)

        followerDirVec = rotateVector(np.array([1, 0]), self.follower.direction)
        followerRightDir = self.follower.direction + 90
        if followerRightDir >= 360:
            followerRightDir -= 360
        followerRightVec = rotateVector(np.array([1, 0]), followerRightDir)
        """
        distances_follower_to_leadhistory = np.linalg.norm(vecs_follower_to_leadhistory, axis=1)
        angles_history_to_dir = calculateAngle(np.array([self.leader.position-self.follower.position, self.leader.position, self.follower.position]), followerDirVec)
        angles_history_to_right = calculateAngle(np.array([self.leader.position-self.follower.position, self.leader.position, self.follower.position]), followerRightVec)
        """
        radar_values = np.zeros(self.radar_sectors_number, dtype=np.float32)
        if len(self.leader_positions_hist) > 0:
            closest_dots = np.array(self.leader_positions_hist[:min(len(self.leader_positions_hist), self.framestack)])
            vecs_follower_to_leadhistory_close = closest_dots - obs['numerical_features'][5:7]
            distances_follower_to_closestDots = np.linalg.norm(vecs_follower_to_leadhistory_close, axis=1)
            angles_history_to_dir = calculateAngle(vecs_follower_to_leadhistory_close, followerDirVec)
            angles_history_to_right = calculateAngle(vecs_follower_to_leadhistory_close, followerRightVec)
            angles_history_to_right[angles_history_to_dir > np.pi / 2] = -angles_history_to_right[
                angles_history_to_dir > np.pi / 2]
            for i in range(self.radar_sectors_number):
                secrot_dots_distances = distances_follower_to_closestDots[
                    (angles_history_to_right >= self.sectorsAngle * i) & (
                                angles_history_to_right < self.sectorsAngle * (i + 1))]
                if len(secrot_dots_distances) > 0:
                    radar_values[i] = np.min(secrot_dots_distances)

        radar_values = np.clip(radar_values / (self.max_distance * 2), -1, 1)
        obs["wrapper_vecs"] = vecs_follower_to_leadhistory_far
        obs["wrapper_radar"] = radar_values
        return obs

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.leader_positions_hist.append(observation['numerical_features'][:2])
        norms = np.linalg.norm(np.array(self.leader_positions_hist) - observation['numerical_features'][5:7], axis=1)
        indexes = np.nonzero(norms <= max(self.follower.width, self.follower.height))[0]
        for index in sorted(indexes, reverse=True):
            del self.leader_positions_hist[index]

        return self.observation(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.leader_positions_hist = list()
        return self.observation(observation)
