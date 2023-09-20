import gym
from collections import deque
import numpy as np
from gym import Wrapper, ObservationWrapper
from gym.spaces import Box
from warnings import warn
try:
    import cv2
except:
    warn("cv2 is not loaded, wrappers with images won't work correctly")
try:
    from continuous_grid_arctic.utils.misc import rotateVector, calculateAngle
except:
    from src.continuous_grid_arctic.utils.misc import rotateVector, calculateAngle


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
        self.use_prev_obs = use_prev_obs
        self.max_prev_obs = max_prev_obs

        for sensor_name, sensor_config in env.follower_sensors.items():
            if sensor_name =='LeaderTrackDetector_vector' or sensor_config.get("sensor_class", None) =='LeaderTrackDetector_vector':
                features_number += sensor_config['position_sequence_length'] * 2

            if sensor_name =='LeaderTrackDetector_radar' or sensor_config.get("sensor_class", None) =='LeaderTrackDetector_radar':
                features_number += sensor_config['radar_sectors_number']
            if sensor_name =='FollowerInfo' or sensor_config.get("sensor_class", None) =='FollowerInfo':
                if 'speed_direction_param' in sensor_config:
                    features_number += sensor_config['speed_direction_param']
            if sensor_name =='LaserSensor' or sensor_config.get("sensor_class", None) =='LaserSensor':
                if sensor_config['return_all_points']:
                    lidar_points_number = (int(sensor_config['available_angle'] / sensor_config['angle_step'])+1) * sensor_config['points_number']
                else:
                    lidar_points_number = (int(sensor_config['available_angle'] / sensor_config['angle_step'])+1)
                if sensor_config["return_only_distances"]:
                    features_number += lidar_points_number
                else:
                    features_number += lidar_points_number * 2

            if sensor_name =='LeaderCorridor_lasers' or sensor_config.get("sensor_class", None) =='LeaderCorridor_lasers':
                if 'front_lasers_count' in sensor_config:
                    features_number += sensor_config['front_lasers_count']
                else:
                    features_number += 3  # env.follower_sensors['LeaderCorridor_lasers']['lasers_count']
                if 'back_lasers_count' in sensor_config:
                    features_number += sensor_config['back_lasers_count']
            if sensor_name in ["LeaderCorridor_lasers_v2"] or sensor_config.get("sensor_class", None) in ["LeaderCorridor_lasers_v2"]:
                features_number += sensor_config['lasers_count']

        self.features_number_num = features_number
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

        for sensor_name in self.follower_sensors.keys():
            sensor_config = self.follower_sensors[sensor_name]
            if sensor_name == 'LeaderTrackDetector_vector' or sensor_config["sensor_class"] == 'LeaderTrackDetector_vector':
                history_vecs = obs[sensor_name].flatten()
                history_vecs = np.clip(history_vecs / self.max_distance, -1, 1)
                features_list.append(history_vecs)
            if sensor_name == 'LeaderTrackDetector_radar' or sensor_config["sensor_class"] == 'LeaderTrackDetector_radar':
                history_radar = obs[sensor_name]
                history_radar = np.clip(history_radar / self.max_distance, 0, 1)
                features_list.append(history_radar)
            if sensor_name == 'LeaderCorridor_lasers' or sensor_config["sensor_class"] == 'LeaderCorridor_lasers':
                corridor_lasers = obs[sensor_name]
                corridor_lasers =  np.clip(corridor_lasers / self.follower.sensors[sensor_name].laser_length, 0, 1)
                features_list.append(corridor_lasers)
            if sensor_name == 'LeaderCorridor_lasers_v2' or sensor_config["sensor_class"] == 'LeaderCorridor_lasers_v2':
                corridor_lasers_v2 = obs[sensor_name]
                corridor_lasers_v2 = np.clip(corridor_lasers_v2 / self.follower.sensors[sensor_name].laser_length, 0, 1)
                features_list.append(corridor_lasers_v2)
            if sensor_name == 'FollowerInfo' or sensor_config["sensor_class"] == 'FollowerInfo':
                follower_info = obs[sensor_name]
                follower_info = np.clip(follower_info, -1, 1)
                features_list.append(follower_info)
            if sensor_name == 'LaserSensor' or sensor_config["sensor_class"] == 'LaserSensor':
                lidar_sensed_points = obs[sensor_name]
                # переход к относительным координатам лучше делать в сенсоре
                if len(lidar_sensed_points.shape)==2:
                    lidar_sensed_points = np.concatenate(lidar_sensed_points)
                lidar_sensed_points = np.clip(lidar_sensed_points / (self.follower.sensors[sensor_name].range * self.PIXELS_TO_METER), -1, 1)
                features_list.append(lidar_sensed_points)
            self.observations_list = np.concatenate(features_list)
        return self.observations_list

    def step(self, action):
        if self.action_values_range is not None:
            action -= self.min
            action /= self.scale
        obs, rews, dones, infos = self.env.step(action)
        obs = self.observation(obs)
        return obs, rews, dones, infos


class ContinuousObserveModifier_sensorPrev(ObservationWrapper):
    """
    A wrapper for sensors that return a history of values. That is, they are expected to show signs of
    form (steps, dimensions)
    """
    def __init__(self, env, action_values_range=None, lz4_compress=False, max_prev_obs=0):
        super().__init__(env)
        self.observations_list = None
        features_number = 0
        self.max_prev_obs = max_prev_obs

        for sensor_name, sensor_config in env.follower_sensors.items():
            if sensor_config["sensor_class"] in ["LeaderCorridor_Prev_lasers_v2", "LeaderCorridor_Prev_lasers_v3"]:
                if sensor_config["pad_sectors"]:
                    features_number += (4*sensor_config['lasers_count'])
                else:
                    features_number += sensor_config['lasers_count']
            if sensor_config["sensor_class"]=="LeaderCorridor_lasers_compas":
                features_number += (5*sensor_config['lasers_count'])

        self.features_number_num = features_number
        self.observation_space = Box(-np.ones([self.max_prev_obs, features_number]),
                                     np.ones([self.max_prev_obs, features_number]))

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

        for sensor_name in self.follower_sensors.keys():
            sensor_config = self.follower_sensors[sensor_name]
            if sensor_name in ['LeaderCorridor_Prev_lasers_v2', "LeaderCorridor_Prev_lasers_v3"] or sensor_config["sensor_class"] in ['LeaderCorridor_Prev_lasers_v2', "LeaderCorridor_Prev_lasers_v3"]:
                corridor_obs = obs[sensor_name]
                assert len(corridor_obs.shape) == 2
                assert corridor_obs.shape[0] == self.max_prev_obs
                corridor_obs = np.clip(corridor_obs / self.follower.sensors[sensor_name].laser_length, 0, 1)
                features_list.append(corridor_obs)
            if sensor_name == 'LeaderCorridor_lasers_compas' or sensor_config["sensor_class"] == 'LeaderCorridor_lasers_compas':
                corridor_obs = obs[sensor_name]
                assert len(corridor_obs.shape) == 2
                assert corridor_obs.shape[0] == self.max_prev_obs
                corridor_obs = np.clip(corridor_obs / self.follower.sensors[sensor_name].laser_length, 0, 1)
                features_list.append(corridor_obs)
        self.observations_list = np.concatenate(features_list, axis=1)
        return self.observations_list

class ContinuousObserveModifierPrev(ObservationWrapper):
    """
    Wrapper for accumulating previous values of two upgraded sensors (1) Beam sensor with 12 beams for the corridor
    and obstacles; 2) Beam sensor for obstacles with 30 (variably) beams
    """

    def __init__(self, env, action_values_range=None, lz4_compress=False):
        super().__init__(env)
        self.observations_list = None
        features_number = 0
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

        return self.observations_list

    def step(self, action):
        if self.action_values_range is not None:
            action -= self.min
            action /= self.scale
        obs, rews, dones, infos = self.env.step(action)
        obs = self.observation(obs)
        return obs, rews, dones, infos

def areDotsOnLeft(line, dots):
    """
    Determine whether the dots lie to the left of the straight line
    line: ndarray [[x1, y1], [x2,y2]]
    dots: ndarray (points, coordinates)
    """
    # D = (x2 - x1) * (yp - y1) - (xp - x1) * (y2 - y1)
    d = (line[1,0] - line[0,0]) * (dots[:,1] - line[0,1]) - (dots[:, 0] - line[0,0]) * (line[1,1]-line[0,1])
    return d > 0.01

# Враппер, который выходы лидара преобразует в 2д картинку
class ContinuousObserveModifier_lidarMap2d(ContinuousObserveModifier_v0):
    def __init__(self,
        env,
        action_values_range=None,
        map_wrapper_forgetting_rate=None,
        resized_image_shape=(84,84),
        add_safezone_on_map=False,
        fill_safe_zone=True,
        lz4_compress=False):
        super().__init__(env)
        self.map_wrapper_forgetting_rate = map_wrapper_forgetting_rate
        self.lidar_angle_steps_count = 1 + self.follower_sensors['LaserSensor']['available_angle'] // self.follower_sensors['LaserSensor']['angle_step']
        self.lidar_points_number = self.follower_sensors['LaserSensor']['points_number']
        self.lidar_range_pixels = self.follower_sensors['LaserSensor']['sensor_range'] * env.PIXELS_TO_METER
        self.resized_image_shape = resized_image_shape
        self.add_safezone_on_map = add_safezone_on_map

        self.observation_space = Box(-np.zeros(list(resized_image_shape) + [3]),
                                     np.ones(list(resized_image_shape) + [3]))
        self.action_values_range = action_values_range
        if self.action_values_range is not None:
            low_bound, high_bound = self.action_values_range
            self.scale = (high_bound - low_bound) / (env.action_space.high - env.action_space.low)
            self.min = low_bound - env.action_space.low * self.scale
            self.action_space = Box(low=-np.ones_like(env.action_space.low),
                                    high=np.ones_like(env.action_space.high),
                                    shape=env.action_space.shape,
                                    dtype=env.action_space.dtype)
        self.prev_lidar_map = None
        self.fill_safe_zone = fill_safe_zone

    def DrawLidar2dMap(self, lidar_map_size,
                     leader_position,
                     follower_position,
                     angle_between_leader_and_follower,
                     lidar_observation,
                     lidar_angle_step,
                     leader_directions,
                     follower_direction):
        """
        leader_position
        follower_position
        """

        # рисуем карту из нулей
        lidar_map = np.zeros((lidar_map_size[0], lidar_map_size[1], 3), dtype=np.float32)
        # Заполняем красный канал единицами - препятствия, потом будем обнулять точки, которые видит лидар
        lidar_map[:,:,0] = 1

        # добавить прямоугольник за лидером
        # вычисляем линии простого прямоугольника за спиной у лидера в относительных координатах
        if self.add_safezone_on_map:
            min_distance, max_distance, max_dev = self.min_distance, self.max_distance, self.max_dev
            rectangle_points = [np.array([-min_distance, max_dev]), np.array([-min_distance, -max_dev]),
                               np.array([-max_distance, max_dev]), np.array([-max_distance, -max_dev])]
            rotated_rectangle_points = [rotateVector(x, leader_directions) for x in rectangle_points]
            actual_rectangle_points = [x+leader_position for x in rotated_rectangle_points]
            relative_to_follower_rectangle_points = [x-follower_position for x in actual_rectangle_points]
            relative_to_follower_rectangle_points_rotated = [rotateVector(x, -follower_direction) for x in relative_to_follower_rectangle_points]
            # проверяем точки лидара на то, находятся ли они внутри этого прямоугольника или нет.

            line = np.array([relative_to_follower_rectangle_points[1], relative_to_follower_rectangle_points[0]])
            insideDots_currRectangle = areDotsOnLeft(line, lidar_observation)
            line = np.array([relative_to_follower_rectangle_points[3], relative_to_follower_rectangle_points[1]])
            insideDots_currRectangle &= areDotsOnLeft(line, lidar_observation)
            line = np.array([ relative_to_follower_rectangle_points[2], relative_to_follower_rectangle_points[3]])
            insideDots_currRectangle &= areDotsOnLeft(line, lidar_observation)
            line = np.array([relative_to_follower_rectangle_points[0], relative_to_follower_rectangle_points[2]])
            insideDots_currRectangle &= areDotsOnLeft(line, lidar_observation)
        else:
            insideDots_currRectangle = np.zeros(lidar_observation.shape[0], dtype=np.bool)
        step_i, point_i = 0, 0
        # идём по точкам, которые вернул лидар
        # предполагается, что каждая точка [0,0] в списке - это начало нового луча
        for x_i, x in enumerate(lidar_observation):
            if (x==0).all():
                point_i = 0
                if x_i !=0:
                    if step_i==0:
                        step_i += 1
                    elif step_i > 0:
                        step_i *= -1
                    elif step_i < 0:
                        step_i *= -1
                        step_i += 1
            # Если эта точка есть в списке, ставим ей 0 в красном канале
            lidar_map[step_i, point_i, 0] = 0
            if insideDots_currRectangle[x_i]:
                lidar_map[step_i, point_i, 2] = 1
            point_i +=1
        if not self.fill_safe_zone:
            #raise NotImplementedError("Не работает. Херня какая-то рисуется")
            safe_zone_border = np.zeros_like(lidar_map[:,:,2])
            for i in range(lidar_map.shape[0]):
                for j in range(lidar_map.shape[1]-1):
                    if lidar_map[i,j,2]==1:
                        if i==0:
                            if j==0:
                                if lidar_map[i,j+1,2]==1 and lidar_map[i+1,j,2]==1 and lidar_map[-1,j,2]==1:
                                    continue
                            if lidar_map[i,j+1,2]==1 and lidar_map[i+1,j,2]==1 and lidar_map[i,j-1,2]==1 and lidar_map[-1,j,2]==1:
                                continue
                        elif i==lidar_map.shape[0]-1:
                            if j==0:
                                if lidar_map[i,j+1,2]==1 and lidar_map[0,j,2]==1 and lidar_map[i-1,j,2]==1:
                                    continue
                            if lidar_map[i,j+1,2]==1 and lidar_map[0,j,2]==1 and lidar_map[i,j-1,2]==1 and lidar_map[i-1,j,2]==1:
                                continue
                        elif j==0:
                            if lidar_map[i,j+1,2]==1 and lidar_map[i+1,j,2]==1 and lidar_map[i-1,j,2]==1:
                                continue
                        elif lidar_map[i,j+1,2]==1 and lidar_map[i+1,j,2]==1 and lidar_map[i,j-1,2]==1 and lidar_map[i-1,j,2]==1:
                            continue
                        else:
                            safe_zone_border[i,j] = 1
            lidar_map[:, :, 2] = safe_zone_border

        # лидеру на карте ставим 1 в зелёном канале
        follower_to_leader_vec = leader_position - follower_position
        follower_to_leader_dist = np.linalg.norm(follower_to_leader_vec)
        distance_between_points = self.lidar_range_pixels / self.lidar_points_number
        if follower_to_leader_dist <= self.lidar_range_pixels:
            leader_row_on_map = -int(angle_between_leader_and_follower // lidar_angle_step)
            leader_col_on_map = int(follower_to_leader_dist // distance_between_points)
            lidar_map[leader_row_on_map, leader_col_on_map, 1] = 1
        return lidar_map

    def observation(self, obs):
        leader_position = np.array([obs['numerical_features'][0], obs['numerical_features'][1]])
        follower_position = np.array([obs['numerical_features'][5], obs['numerical_features'][6]])
        relative_leader_position = leader_position - follower_position
        relative_leader_position_2 = rotateVector(relative_leader_position, -obs['numerical_features'][8])
        arccos_x = np.arccos(relative_leader_position_2.dot(np.array([1, 0])) / (np.linalg.norm(relative_leader_position_2) * np.linalg.norm(np.array([1, 0]))))
        arccos_y = np.arccos(relative_leader_position_2.dot(np.array([0, 1])) / (np.linalg.norm(relative_leader_position_2) * np.linalg.norm(np.array([0, 1]))))
        if arccos_y > np.pi / 2:
            arccos_x = -arccos_x
        angle_between_leader_and_follower = np.degrees(arccos_x)

        lidar_map = self.DrawLidar2dMap((self.lidar_angle_steps_count, self.lidar_points_number), leader_position, follower_position,
                                 angle_between_leader_and_follower,  obs["LaserSensor"], self.follower_sensors['LaserSensor']['angle_step'],
                                 obs['numerical_features'][3], obs['numerical_features'][8])

        lidar_map = np.roll(lidar_map, self.lidar_angle_steps_count // 2, axis=0)
        if self.prev_lidar_map is not None:
            lidar_map += (self.prev_lidar_map - self.map_wrapper_forgetting_rate)
            lidar_map = np.clip(lidar_map, 0, 1)
        self.prev_lidar_map = lidar_map
        resized = cv2.resize(lidar_map, self.resized_image_shape, interpolation = cv2.INTER_NEAREST)
        return resized

    def step(self, action):
        if self.action_values_range is not None:
            action -= self.min
            action /= self.scale
        obs, rews, dones, infos = self.env.step(action)
        obs = self.observation(obs)
        return obs, rews, dones, infos

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.prev_lidar_map = None

        return self.observation(observation)

class ContinuousObserveModifier_lidarMap2d_v2(ContinuousObserveModifier_lidarMap2d):
    def __init__(self,
                 env,
                 action_values_range=None,
                 map_wrapper_forgetting_rate=None,
                 resized_image_shape=(84,84),
                 add_safezone_on_map=False,
                 saving_leader_history_period=8,
                 fill_safe_zone=True,
                 lz4_compress=False):
        super().__init__(env, action_values_range,
                 map_wrapper_forgetting_rate,
                 resized_image_shape,
                 add_safezone_on_map,
                 fill_safe_zone,
                 lz4_compress)
        self.leader_positions_hist = deque()
        self.corridor = deque()
        self.saving_counter = 0
        self.saving_period = saving_leader_history_period
        self.prev_follower_position = None
        self.prev_follower_direction = None

    def DrawLidar2dMap(self, lidar_map_size,
                     leader_position,
                     follower_position,
                     angle_between_leader_and_follower,
                     lidar_observation,
                     lidar_angle_step,
                     leader_directions,
                     follower_direction):
        """
        leader_position
        follower_position
        """

        # рисуем карту из нулей
        lidar_map = np.zeros((lidar_map_size[0], lidar_map_size[1], 3), dtype=np.float32)
        # Заполняем красный канал единицами - препятствия, потом будем обнулять точки, которые видит лидар
        lidar_map[:,:,0] = 1

        # добавить прямоугольник за лидером
        # вычисляем линии простого прямоугольника за спиной у лидера в относительных координатах
        if self.add_safezone_on_map:
            dotsInsideSafeZone = None
            if len(self.corridor)>2:
                for i in range(len(self.corridor)-1):
                    rectangle_points = [self.corridor[i][0], self.corridor[i][1],
                                       self.corridor[i+1][0], self.corridor[i+1][1]]
                    check1 = areDotsOnLeft(np.array([rectangle_points[0], rectangle_points[3]]), np.array([rectangle_points[1], rectangle_points[2]]))
                    check2 = areDotsOnLeft(np.array([rectangle_points[1], rectangle_points[2]]), np.array([rectangle_points[3], rectangle_points[0]]))
                    if ((check1==[False, True]).all() and (check2==[False, True]).all()):
                        pass
                    elif ((check1==[True, True]).all() and (check2==[True, True]).all()):
                        rectangle_points[1], rectangle_points[3] = rectangle_points[3], rectangle_points[1]
                    elif ((check1==[False, False]).all() and (check2==[False, False]).all()):
                        rectangle_points[0], rectangle_points[2] = rectangle_points[2], rectangle_points[0]
                    elif ((check1==[True, True]).all() and (check2==[False, False]).all()):
                        rectangle_points[0], rectangle_points[1] = rectangle_points[1], rectangle_points[0]
                    elif ((check1==[False, False]).all() and (check2==[True, True]).all()):
                        rectangle_points[2], rectangle_points[3] = rectangle_points[3], rectangle_points[2]
                    elif ((check1==[True, False]).all() and (check2==[True, False]).all()):
                        rectangle_points[2], rectangle_points[3] = rectangle_points[3], rectangle_points[2]
                        rectangle_points[0], rectangle_points[1] = rectangle_points[1], rectangle_points[0]
                    elif (((check1==[False, True]).all() or (check1==[True, False]).all()) and (check2==[True, True]).all()):
                        warn("Впуклый прямоугольник, не обрабатывается корректно")
                    elif ((check1==[True, True]).all() and ((check2==[True, False]).all() or (check2==[False, True]).all())):
                        warn("Впуклый прямоугольник, не обрабатывается корректно")
                    elif ((check1==[False, False]).all() and ((check2==[True, False]).all() or (check2==[False, True]).all())):
                        warn("Впуклый прямоугольник, не обрабатывается корректно")
                    elif (((check1==[False, True]).all() or (check1==[True, False]).all()) and (check2==[False, False]).all()):
                        warn("Впуклый прямоугольник, не обрабатывается корректно")
                    else:
                        raise ValueError("Не предвидел такой вариант расположения вершин прямоугольника (сегмента корридора) при проверке, находятся ли точки внутри него: check1:{}, check2:{}".format(str(check1), str(check2)))
                    # проверяем точки лидара на то, находятся ли они внутри этого прямоугольника или нет.
                    # Проверка для каждой стороны 4-ёхугольника, лежат ли точки слева от неё. Точки, которые слева от всех сторон - внутри многоугольника.
                    # TODO: Не работает с впуклыми многоугольниками, возможно стоит попробовать алгоритм с лучами или ещё что-то.
                    line = np.array([rectangle_points[0], rectangle_points[1]])
                    insideDots_currRectangle = areDotsOnLeft(line, lidar_observation)
                    line = np.array([rectangle_points[1], rectangle_points[3]])
                    insideDots_currRectangle &= areDotsOnLeft(line, lidar_observation)
                    line = np.array([rectangle_points[3], rectangle_points[2]])
                    insideDots_currRectangle &= areDotsOnLeft(line, lidar_observation)
                    line = np.array([rectangle_points[2], rectangle_points[0]])
                    insideDots_currRectangle &= areDotsOnLeft(line, lidar_observation)
                    if i==0:
                        dotsInsideSafeZone = insideDots_currRectangle
                    else:
                        dotsInsideSafeZone |= insideDots_currRectangle
            else:
                dotsInsideSafeZone = np.zeros(lidar_observation.shape[0], dtype=np.bool)
        else:
            dotsInsideSafeZone = np.zeros(lidar_observation.shape[0], dtype=np.bool)
        step_i, point_i = 0, 0
        # идём по точкам, которые вернул лидар
        # предполагается, что каждая точка [0,0] в списке - это начало нового луча
        for x_i, x in enumerate(lidar_observation):
            if (x==0).all():
                point_i = 0
                if x_i !=0:
                    if step_i==0:
                        step_i += 1
                    elif step_i > 0:
                        step_i *= -1
                    elif step_i < 0:
                        step_i *= -1
                        step_i += 1
            # Если эта точка есть в списке, ставим ей 0 в красном канале
            lidar_map[step_i, point_i, 0] = 0
            if dotsInsideSafeZone[x_i]:
                lidar_map[step_i, point_i, 2] = 1
            point_i +=1

        if not self.fill_safe_zone:
            raise NotImplementedError("Не работает. Херня какая-то рисуется")
            safe_zone_border = np.zeros_like(lidar_map[:,:,2])
            for i in range(lidar_map.shape[0]):
                for j in range(lidar_map.shape[1]-1):
                    if lidar_map[i,j,2]==1:
                        if i==0:
                            if j==0:
                                if lidar_map[i,j+1,2]==1 and lidar_map[i+1,j,2]==1 and lidar_map[-1,j,2]==1:
                                    continue
                            if lidar_map[i,j+1,2]==1 and lidar_map[i+1,j,2]==1 and lidar_map[i,j-1,2]==1 and lidar_map[-1,j,2]==1:
                                continue
                        elif i==lidar_map.shape[0]-1:
                            if j==0:
                                if lidar_map[i,j+1,2]==1 and lidar_map[0,j,2]==1 and lidar_map[i-1,j,2]==1:
                                    continue
                            if lidar_map[i,j+1,2]==1 and lidar_map[0,j,2]==1 and lidar_map[i,j-1,2]==1 and lidar_map[i-1,j,2]==1:
                                continue
                        elif j==0:
                            if lidar_map[i,j+1,2]==1 and lidar_map[i+1,j,2]==1 and lidar_map[i-1,j,2]==1:
                                continue
                        elif lidar_map[i,j+1,2]==1 and lidar_map[i+1,j,2]==1 and lidar_map[i,j-1,2]==1 and lidar_map[i-1,j,2]==1:
                            continue
                        else:
                            safe_zone_border[i,j] = 1
            lidar_map[:, :, 2] = safe_zone_border
        # лидеру на карте ставим 1 в зелёном канале
        follower_to_leader_vec = leader_position - follower_position
        follower_to_leader_dist = np.linalg.norm(follower_to_leader_vec)
        distance_between_points = self.lidar_range_pixels / self.lidar_points_number
        if follower_to_leader_dist <= self.lidar_range_pixels:
            leader_row_on_map = -int(angle_between_leader_and_follower // lidar_angle_step)
            leader_col_on_map = int(follower_to_leader_dist // distance_between_points)
            lidar_map[leader_row_on_map, leader_col_on_map, 1] = 1
        return lidar_map

    def constructCorridor(self, relative_leader_position, follower_position, follower_direction):
        if self.saving_counter % self.saving_period == 0:
            if self.prev_follower_position is None:
                self.prev_follower_position = follower_position

            follower_delta = follower_position - self.prev_follower_position
            if self.prev_follower_direction is None:
                self.prev_follower_direction = follower_direction
            follower_rotation_delta = follower_direction - self.prev_follower_direction
            # Если позиция лидера не изменилась с последнего обсерва, ничего не обновляем
            if len(self.leader_positions_hist) > 0 and (self.leader_positions_hist[-1] == relative_leader_position).all():
                return
            # Если симуляция только началась, сохраняем текущую ведомого, чтоб начать от неё строить коридор
            if len(self.leader_positions_hist) == 0 and self.saving_counter == 0:
                self.leader_positions_hist.extend(np.array(x) for x in
                                                  zip(np.linspace(0, relative_leader_position[0],
                                                                  10),
                                                      np.linspace(0, relative_leader_position[1],
                                                                  10)))
            else:
                self.leader_positions_hist.append(relative_leader_position.copy())
            # move old leader positions accodring to follower delta
            for i in range(len(self.leader_positions_hist)-1):
                self.leader_positions_hist[i] = self.leader_positions_hist[i] - follower_delta
                self.leader_positions_hist[i] = rotateVector(self.leader_positions_hist[i], -follower_rotation_delta)
            for i in range(len(self.corridor)):
                self.corridor[i][0] = self.corridor[i][0] - follower_delta
                self.corridor[i][1] = self.corridor[i][1] - follower_delta
                self.corridor[i][0] = rotateVector(self.corridor[i][0], -follower_rotation_delta)
                self.corridor[i][1] = rotateVector(self.corridor[i][1], -follower_rotation_delta)

            dists = np.linalg.norm(np.array(self.leader_positions_hist)[:-1, :] -
                                   np.array(self.leader_positions_hist)[1:, :], axis=1)
            path_length = np.sum(dists)
            while path_length > self.max_distance:
                self.leader_positions_hist.popleft()
                if len(self.corridor) > 0:
                    self.corridor.popleft()
                dists = np.linalg.norm(np.array(self.leader_positions_hist)[:-1, :] -
                                       np.array(self.leader_positions_hist)[1:, :], axis=1)
                path_length = np.sum(dists)

            if len(self.leader_positions_hist) > 1:
                if self.saving_counter == 0:
                    for i in range(len(self.leader_positions_hist) - 1, 0, -1):
                        last_2points_vec = self.leader_positions_hist[i] - self.leader_positions_hist[i-1]
                        last_2points_vec *= self.max_dev / np.linalg.norm(last_2points_vec)
                        right_border_dot = rotateVector(last_2points_vec, 90)
                        right_border_dot += self.leader_positions_hist[-i-1]
                        left_border_dot = rotateVector(last_2points_vec, -90)
                        left_border_dot += self.leader_positions_hist[-i-1]
                        self.corridor.append([right_border_dot, left_border_dot])
                last_2points_vec = self.leader_positions_hist[-1] - self.leader_positions_hist[-2]
                last_2points_vec *= self.max_dev / np.linalg.norm(last_2points_vec)
                right_border_dot = rotateVector(last_2points_vec, 90)
                right_border_dot += self.leader_positions_hist[-2]
                left_border_dot = rotateVector(last_2points_vec, -90)
                left_border_dot += self.leader_positions_hist[-2]
                self.corridor.append([right_border_dot, left_border_dot])
            self.prev_follower_position = follower_position
            self.prev_follower_direction = follower_direction
        self.saving_counter += 1

    def observation(self, obs):
        leader_position = np.array([obs['numerical_features'][0], obs['numerical_features'][1]])
        follower_position = np.array([obs['numerical_features'][5], obs['numerical_features'][6]])

        relative_leader_position = leader_position - follower_position
        relative_leader_position_2 = rotateVector(relative_leader_position, -obs['numerical_features'][8])
        arccos_x = np.arccos(relative_leader_position_2.dot(np.array([1, 0])) / (np.linalg.norm(relative_leader_position_2) * np.linalg.norm(np.array([1, 0]))))
        arccos_y = np.arccos(relative_leader_position_2.dot(np.array([0, 1])) / (np.linalg.norm(relative_leader_position_2) * np.linalg.norm(np.array([0, 1]))))
        if arccos_y > np.pi / 2:
            arccos_x = -arccos_x
        angle_between_leader_and_follower = np.degrees(arccos_x)
        self.constructCorridor(relative_leader_position, follower_position, obs['numerical_features'][8])

        lidar_map = self.DrawLidar2dMap((self.lidar_angle_steps_count, self.lidar_points_number), leader_position, follower_position,
                                 angle_between_leader_and_follower,  obs["LaserSensor"], self.follower_sensors['LaserSensor']['angle_step'],
                                 obs['numerical_features'][3], obs['numerical_features'][8])

        lidar_map = np.roll(lidar_map, self.lidar_angle_steps_count // 2, axis=0)
        if self.prev_lidar_map is not None:
            lidar_map += (self.prev_lidar_map - self.map_wrapper_forgetting_rate)
            lidar_map = np.clip(lidar_map, 0, 1)
        self.prev_lidar_map = lidar_map
        resized = cv2.resize(lidar_map, self.resized_image_shape, interpolation = cv2.INTER_NEAREST)
        return resized

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.prev_lidar_map = None
        self.leader_positions_hist.clear()
        self.corridor.clear()
        self.saving_counter = 0
        self.prev_follower_position = None
        self.prev_follower_direction = None

        return self.observation(observation)

class LeaderTrajectory_v0(ObservationWrapper):
    """
    Deprecated class, needed only to check backward compatibility with experiments launched on
    commit 86211bf4a3b0406e23bc561c00e1ea975c20f90b
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
        A vector with 13 components is expected as input:
        - position x of the leader
        - position y of the leader
        - leader speed
        - direction of the leader
        - leader turning speed
        - position x of the agent
        - position y of the agent
        - agent speed
        - agent direction
        - agent rotation speed
        - minimum distance
        - maximum distance
        - maximum deviation from the route

        The output is a vector of 2 concatenated vectors:
        - vector of pairs of coordinates of vectors of differences between the current position of the slave and the
          last N positions of the master through which he passed
        - radar vector, has a number of components equal to the argument from the radar_sectors_number config, each
          component is the distance to the nearest point in the corresponding sector of the semicircle in front of it
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

class SkipBadSeeds(Wrapper):
    """
    Pass wrapper
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)        
        while not self.env.found_target_point:
            observation = self.env.reset(**kwargs)
        return observation