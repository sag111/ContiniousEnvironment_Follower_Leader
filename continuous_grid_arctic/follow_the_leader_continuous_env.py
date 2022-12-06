import os
import random
import time
from warnings import warn
from collections import OrderedDict
import pygame
from math import radians, cos, sin
import numpy as np
import pandas as pd
from scipy.spatial import distance
import gym
from gym.envs.registration import register as gym_register
from gym.spaces import Discrete, Box

from continuous_grid_arctic.utils.classes import AbstractRobot, GameObject, RobotWithSensors
from continuous_grid_arctic.utils.reward_constructor import Reward
from continuous_grid_arctic.utils.astar import astar
from continuous_grid_arctic.utils.misc import angle_correction, angle_to_point, distance_to_rect
from continuous_grid_arctic.utils.rrt_star import RRTStar
from continuous_grid_arctic.utils.lqr_rrt_star import LQRRRTStar
from continuous_grid_arctic.utils.dstar import Map, Dstar
from continuous_grid_arctic.utils.rrt import RRT

try:
    from utils.misc import angle_correction, rotateVector, calculateAngle, distance_to_rect
except:
    from continuous_grid_arctic.utils.misc import angle_correction, rotateVector, calculateAngle, distance_to_rect


# TODO: Вынести все эти дефолтные настройки в дефолтный конфиг, возможно разбить конфиг на подконфиги
# как вариант - файл default_configs, там словари. Они сразу подгружаются средой, если в среду переданы другие словари,
# совпадающие ключи перезаписываются
class Game(gym.Env):
    def __init__(self, game_width=1500,
                 game_height=1000,
                 framerate=500,
                 frames_per_step=10,
                 random_frames_per_step=None,
                 caption="Serious Robot Follower Simulation v.-1",
                 trajectory=None,
                 leader_pos_epsilon=25,
                 show_leader_path=True,
                 show_leader_trajectory=True,
                 show_rectangles=True,
                 show_box=True,
                 show_sensors=True,
                 simulation_time_limit=None,
                 reward_config=None,
                 pixels_to_meter=50,
                 min_distance=1,  # в метрах
                 max_distance=4,  # в метрах
                 max_dev=1,  # в метрах
                 warm_start=500,  # во фреймах
                 manual_control=False,
                 max_steps=5000,
                 aggregate_reward=False,
                 add_obstacles=True,
                 add_bear=True,
                 bear_number = 3,
                 multi_random_bears = False,
                 obstacle_number=35,
                 bear_behind=False,
                 # end_simulation_on_leader_finish=False,  # NotImplemented
                 # discretization_factor=5,  # NotImplemented
                 step_grid=10,
                 early_stopping={},
                 follower_sensors={},
                 leader_speed_regime=None,
                 leader_acceleration_regime=None,
                 discrete_action_space=False,
                 constant_follower_speed=False,
                 path_finding_algorythm="dstar",
                 multiple_end_points=False,
                 corridor_length=4,
                 corridor_width=1,
                 negative_speed=False,
                 follower_speed_koeff=0.5,
                 **kwargs
                 ):
        """Класс, который создаёт непрерывную среду для решения задачи следования за лидером.
        Входные параметры:
        game_width (int): 
            ширина игрового экрана (в пикселях);
        game_height (int): 
            высота игрового экрана (в пикселях);
        framerate (int): ...;
        caption (str): 
            заголовок игрового окна;
        trajectory (list of points or None): 
            список точек, по которым едет Ведущий. Если None, список генерируется случайно;
        leader_pos_epsilon (int): 
            расстояние в пикселях от точки траектории, в пределах которого считается, что Ведущий прошёл через точку;
        show_leader_path (bool): 
            флаг отображения всего маршрута, по которому идёт ведущий;
        show_leader_trajectory (bool): 
            флаг отображения пройденного ведущим маршрута;
        show_rectangles (bool): 
            флаг отображения прямоугольников взаимодействия;
        show_box (bool): 
            флаг отображения границ, в которых нужно находиться Ведомому;
        simulation_time_limit (int or None):
            лимит по времени одной симуляции (в секундах, если None -- не ограничен)
        reward_config (str, Path or None): 
            путь до json-конфигурации награды, созданной с помощью класса reward_constructor. Если None, создаёт reward
            по умолчанию (Ivan v.1)
        pixels_to_meter (int): 
            число пикселей, который в данной модели отражает один метр;
        min_distance (int): 
            минимальная дистанция (в метрах), которую должен соблюдать Ведомый от Ведущего;
        max_distance (int): 
            максимальная дистанция (в метрах), дальше которой Ведомый не должен отставать от Ведущего (по маршруту);
        max_dev (int): 
            максимальная дистанция (в метрах), в пределах которой Ведомый может отклониться от маршрута;
        warm_start (int): 
            число шагов, в пределах которого Ведомый не будет получать штраф (сейчас не реализовано полноценно);
        manual_control (bool): 
            использовать ручное управление Ведомым;
        max_steps (int): 
            максимальное число шагов для одной симуляции;
        aggregate_reward (bool):
            если True, step будет давать акумулированную награду;
        obstacle_number (int):
            число случайно генерируемых препятствий.
        leader_speed_regime (dict):
            словарь - ключ - число степов, значение - скорость лидера;
        leader_
        constant_follower_speed (bool):
            флаг - если True - корость ведомого всегда будет максимальной, и будет использован только один экшн - поворот
        random_frames_per_step (tuple/list):
            диапазон из которого будет сэмплироваться frames_per_step
        number_of_target_points (int):
            количество точек через которые будет строится маршрут. По умолчанию одна целевая.
        path_finding_algorythm (str):
            какой алгоритм поиска пути использовать для лидера. astar, dstar
        multiple_end_points (bool):
            если False - используется только одна конечная точка, если True - генерируется несколько точек и более сложный маршрут
        """

        # нужно для сохранения видео
        self.metadata = {"render.modes": ["rgb_array"]}
        # Здесь можно задать дополнительные цвета в формате RGB
        self.colours = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'gray': (30, 30, 30),
            'blue': (0, 0, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            "pink": (251, 204, 231)
        }
        self.early_stopping = early_stopping
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 30)

        # TODO: сделать нормально
        metadata = {"render.modes": ["human", "rgb_array"],
                    "video.frames_per_second": framerate}  # "human" вроде не обязательно
        self.constant_follower_speed = constant_follower_speed
        self.path_finding_algorythm = path_finding_algorythm

        # задание траектории, которое полноценно обрабатывается в методе reset()
        self.trajectory = trajectory
        self.trajectory_generated = False

        self.step_grid = step_grid
        self.accumulated_penalty = 0  # штраф, который копится, если долго не получаем отрицательной награды
        # Генерация финишной точки
        self.finish_point = (10, 10)  # np.float64((random.randrange(20, 100, 10), random.randrange(20, 1000, 10)))
        self.multiple_end_points = multiple_end_points
        if multiple_end_points:
            if path_finding_algorythm != "dstar":
                raise NotImplementedError("Only dstar pathfinding function supports multiple end points. "
                                          "multiple_end_points must be False or diggerent path_finding_algorythm "
                                          "must be chosen")
            # np.float64((random.randrange(20, 100, 10), random.randrange(50, 1000, 10)))
            self.finish_point2 = (1490, 990)
            # np.float64((random.randrange(520, 100, 10), random.randrange(720, 1000, 10)))
            self.finish_point3 = (0, 990)

        # номер симуляции
        self.simulation_number = 0

        self.DISPLAY_WIDTH = game_width
        self.DISPLAY_HEIGHT = game_height
        self.PIXELS_TO_METER = pixels_to_meter
        self.framerate = framerate
        self.frames_per_step = frames_per_step
        self.random_frames_per_step = random_frames_per_step
        # если частота сохранения точек пути совпадает с частотой обсёрвов,
        # сенсор фолловера может брать траекторию прямо из среды, иначе нет
        # пока что он сохраняет траекторию сам, по обсёрвам. Надо подумать,
        # могут ли отличаться частота сохранения и частота обсёрвов.
        self.trajectory_saving_period = 5

        self.leader_pos_epsilon = leader_pos_epsilon

        # Настройки визуализации
        self.show_leader_path = show_leader_path
        self.show_leader_trajectory = show_leader_trajectory
        self.show_rectangles = show_rectangles
        self.show_box = show_box
        self.show_sensors = show_sensors

        self.simulation_time_limit = simulation_time_limit

        if reward_config:
            self.reward_config = Reward.from_json(reward_config)
        else:
            self.reward_config = Reward(leader_movement_reward=0)

        self.overall_reward = 0

        self.min_distance = self._to_pixels(min_distance)
        self.max_distance = self._to_pixels(max_distance)
        self.max_dev = self._to_pixels(max_dev)

        self.warm_start = warm_start

        self.leader_img = pygame.image.load("{}/imgs/car_yellow.png".format(os.path.dirname(os.path.abspath(__file__))))
        self.follower_img = pygame.image.load(
            "{}/imgs/car_poice.png".format(os.path.dirname(os.path.abspath(__file__))))
        self.wall_img = pygame.image.load("{}/imgs/wall.png".format(os.path.dirname(os.path.abspath(__file__))))
        self.rock_img = pygame.image.load("{}/imgs/rock.png".format(os.path.dirname(os.path.abspath(__file__))))

        self.bear_img = pygame.image.load("{}/imgs/bear.png".format(os.path.dirname(os.path.abspath(__file__))))

        self.caption = caption
        self.manual_control = manual_control
        self.max_steps = max_steps
        self.aggregate_reward = aggregate_reward

        self.add_obstacles = add_obstacles
        self.add_bear = add_bear
        self.bear_number = bear_number
        self.multi_random_bears = multi_random_bears
        self.bear_behind = bear_behind
        self.corridor_length = self._to_pixels(corridor_length)
        self.corridor_width = self._to_pixels(corridor_width)
        self.negative_speed = negative_speed
        self.follower_speed_koeff = follower_speed_koeff
        # TODO : _____
        self.obstacles = list()
        self.obstacle_number = obstacle_number

        if not self.add_obstacles:
            self.obstacle_number = 0

        self.follower_sensors = follower_sensors
        self.finish_position_framestimer = None
        # TODO: вынести куда-то дефолтный конфиг, и загружать его
        # TODO : конфиг для отрицательной скорости, наверное стоит поправить это
        if self.negative_speed:
            self.follower_config = {
                # 'min_speed': 0,
                'min_speed':-(self._to_pixels(self.follower_speed_koeff) / 100),
                'max_speed': self._to_pixels(self.follower_speed_koeff) / 100,
                'max_rotation_speed': 57.296 / 100,
            }
        else:
            self.follower_config = {
                'min_speed': 0,
                # 'min_speed':-(self._to_pixels(0.5) / 100),
                'max_speed': self._to_pixels(self.follower_speed_koeff) / 100,
                'max_rotation_speed': 57.296 / 100,
            }
        self.leader_config = {
            'min_speed': 0,
            'max_speed': self._to_pixels(0.5) / 100,
            'max_rotation_speed': 57.296 / 100,
        }
        self.discrete_action_space = discrete_action_space

        if self.discrete_action_space:
            self.action_space = Discrete(5)

            self.discrete_rotation_speed_to_value = {0: -self.follower_config['max_rotation_speed'],
                                                     1: -self.follower_config['max_rotation_speed'] / 2,
                                                     2: 0,
                                                     3: self.follower_config['max_rotation_speed'] / 2,
                                                     4: self.follower_config['max_rotation_speed']}
        elif self.constant_follower_speed:
            self.action_space = Box(
                low=-self.follower_config['max_rotation_speed'], high=self.follower_config['max_rotation_speed'],
                shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = Box(
                np.array((self.follower_config['min_speed'], -self.follower_config['max_rotation_speed']),
                         dtype=np.float32),
                np.array((self.follower_config['max_speed'], self.follower_config['max_rotation_speed']),
                         dtype=np.float32))

        self._create_observation_space()
        # Скорость лидера
        self.leader_speed_regime = None
        if type(leader_speed_regime) in [dict, OrderedDict]:
            self.leader_speed_regime = {}
            for k, v in leader_speed_regime.items():
                self.leader_speed_regime[int(k)] = v
        elif leader_speed_regime is not None:
            warn("leader_speed_regime должен быть dict или OrderedDict, получено: {}, будет проигнорировано".format(
                type(leader_speed_regime)))
        self.leader_acceleration_regime = None
        if type(leader_acceleration_regime) in [dict, OrderedDict]:
            self.leader_acceleration_regime = {}
            for k, v in leader_acceleration_regime.items():
                self.leader_acceleration_regime[int(k)] = v
        elif leader_acceleration_regime is not None:
            warn("leader_acceleration_regime должен быть dict, получено: {}, будет проигнорировано".format(
                type(leader_acceleration_regime)))

        if random_frames_per_step is not None and frames_per_step is not None:
            warn(
                "Одновременно заданы и random_frames_per_step и frames_per_step, будет использоваться random_frames_per_step")
            assert len(
                random_frames_per_step) == 2, "random frames per step должен быть задан в виде границ для генерации случайных значений. Задано: {}".format(
                random_frames_per_step)
            self.frames_per_step = np.random.randint(random_frames_per_step[0], random_frames_per_step[1])
        self.check_parameters()
        self.green_zone_trajectory_points = list()
        self.left_border_points_list = list()
        self.right_border_points_list = list()
        self.follower_scan_dict = {}
        self.step_count = 0
        self.cur_speed_multiplier = 1
        self.game_object_list = list()
        self.game_dynamic_list = list()


    def check_parameters(self):
        """
        Проверка параметров на допустимые значения.
        """
        if self.path_finding_algorythm not in ["astar", "dstar"]:
            raise ValueError(
                "path_finding_algorythm {} not in list:{}".format(self.path_finding_algorythm, ["astar", "dstar"]))

    def seed(self, seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        return

    def reset(self):
        """Стандартный для gym обработчик инициализации новой симуляции. Возвращает инициирующее наблюдение."""

        print("===Запуск симуляции номер {}===".format(self.simulation_number))
        self.step_count = 0
        self.accumulated_penalty = 0
        self.cur_speed_multiplier = 1

        #         valid_trajectory = False

        # Список всех игровых объектов
        self.game_object_list = list()

        # Список всех динамических препятствий
        self.game_dynamic_list = list()

        # Создание ведущего и ведомого
        self._create_robots()

        # Создание препятствий
        if self.add_obstacles:
            self._create_obstacles()


        # в случае, если траектория не задана или была сгенерирована, при каждой симуляции генерируем новую
        # случайную траекторию
        if (self.trajectory is None) or self.trajectory_generated:
            self.finish_point = self.generate_finish_point([20, 20], [int(self.DISPLAY_WIDTH/2), self.DISPLAY_HEIGHT - 20])
            if self.multiple_end_points:
                if self.finish_point[1] >= (self.DISPLAY_HEIGHT / 2):
                    self.finish_point2 = self.generate_finish_point([20, 20], [self.DISPLAY_WIDTH-20, int(self.DISPLAY_HEIGHT / 2)])
                else:
                    self.finish_point2 = self.generate_finish_point([20, int(self.DISPLAY_HEIGHT / 2)],  [self.DISPLAY_WIDTH-20, self.DISPLAY_HEIGHT - 20])
                if self.finish_point2[1] >= (self.DISPLAY_HEIGHT / 2):
                    self.finish_point3 = self.generate_finish_point([20, 20], [self.DISPLAY_WIDTH - 20,
                                                                               int(self.DISPLAY_HEIGHT / 2)])
                else:
                    self.finish_point3 = self.generate_finish_point([20, int(self.DISPLAY_HEIGHT / 2)],
                                                                    [self.DISPLAY_WIDTH - 20, self.DISPLAY_HEIGHT - 20])
            if self.path_finding_algorythm == "dstar":
                self.trajectory = self.generate_trajectory_dstar()
            elif self.path_finding_algorythm == "astar":
                self.trajectory = self.generate_trajectory_astar(max_iter=None)
            self.trajectory_generated = True

        # TODO : перенести в конфиг
        if self.add_bear:
            self._create_dyn_obs()

        # список точек пройденного пути Ведущего, которые попадают в границы требуемого расстояния
        self.green_zone_trajectory_points = list()
        self.left_border_points_list = list()
        self.right_border_points_list = list()

        # Флаг конца симуляции
        self.done = False

        # Флаги для расчёта reward
        self._init_reward_flags()
        self.overall_reward = 0

        self.cur_target_id = 1  # индекс целевой точки из маршрута

        self.leader_finished = False  # флаг, показывает, закончил ли лидер маршрут, т.е. достиг ли последней точки
        if len(self.trajectory) == 0:
            self.done = True
            self.cur_target_point = self.leader.start_position
        else:
            self.cur_target_point = self.trajectory[
                self.cur_target_id]  # координаты текущей целевой точки (возможно избыточны)

        # Инициализация сеанса pygame, создание окна и часов
        pygame.init()
        self.gameDisplay = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        pygame.display.set_caption(self.caption)
        self.clock = pygame.time.Clock()

        self.simulation_number += 1

        # располагаем ведомого с учётом того, куда направлен лидер
        self.leader.direction = angle_to_point(self.leader.position, self.cur_target_point)
        self._pos_follower_behind_leader()

        # TODO: позиционирование препятствий
        if self.bear_behind:
            # self._pos_bears_nearest_leader()
            self._reset_pose_bear()



        self.leader_factual_trajectory = list()  # список, который сохраняет пройденные лидером точки;
        # добавляем начальные позиции - от ведомого до лидера, чтоб там была сейф зона.
        first_dots_for_follower_count = int(distance.euclidean(self.follower.position, self.leader.position) / (
                self.trajectory_saving_period * self.leader.max_speed))
        self.leader_factual_trajectory.extend(
            zip(np.linspace(self.follower.position[0], self.leader.position[0], first_dots_for_follower_count),
                np.linspace(self.follower.position[1], self.leader.position[1], first_dots_for_follower_count)))

        self.follower_scan_dict = self.follower.use_sensors(self)
        self.finish_position_framestimer = None
        return self._get_obs()

    def _create_robots(self):
        # TODO: сторонние конфигурации для создания роботов
        #  TODO : исправить (уйти от привязки к переменной self.max_distance)
        leader_start_position = (
            random.randrange(self.DISPLAY_WIDTH / 2 + self.max_distance, self.DISPLAY_WIDTH - self.max_distance, 10),
            random.randrange(self.max_distance, self.DISPLAY_HEIGHT - self.max_distance, 10))

        leader_start_direction = angle_to_point(leader_start_position,
                                                np.array((self.DISPLAY_WIDTH / 2, self.DISPLAY_HEIGHT / 2),
                                                         dtype=int))  # random.randint(1,360)

        self.leader = AbstractRobot("leader",
                                    image=self.leader_img,
                                    height=self._to_pixels(0.38),
                                    width=self._to_pixels(0.52),
                                    min_speed=self.leader_config["min_speed"],
                                    max_speed=self.leader_config["max_speed"],
                                    max_speed_change=self._to_pixels(0.005),  # / 100,
                                    max_rotation_speed=self.leader_config["max_rotation_speed"],
                                    max_rotation_speed_change=20 / 100,
                                    start_position=leader_start_position,
                                    start_direction=leader_start_direction)

        # !!! вся эта процедура повторяется после создания в резете при вызове _pos_follower_behind_leader
        follower_start_distance_from_leader = random.randrange(int(self.min_distance * 1.1),
                                                               int(self.max_distance * 0.9), 1)
        follower_start_position_theta = radians(angle_correction(leader_start_direction + 180))
        follower_start_position = np.array((follower_start_distance_from_leader * cos(follower_start_position_theta),
                                            follower_start_distance_from_leader * sin(
                                                follower_start_position_theta))) + leader_start_position

        follower_direction = angle_to_point(follower_start_position, self.leader.position)

        self.follower = RobotWithSensors("follower",
                                         image=self.follower_img,
                                         start_direction=follower_direction,
                                         height=self._to_pixels(0.5),
                                         width=self._to_pixels(0.35),
                                         min_speed=self.follower_config["min_speed"],
                                         max_speed=self.follower_config["max_speed"],
                                         max_speed_change=self._to_pixels(0.005) / 100,
                                         max_rotation_speed=self.follower_config["max_rotation_speed"],
                                         max_rotation_speed_change=20 / 100,
                                         start_position=follower_start_position,
                                         sensors=self.follower_sensors)

        self.cur_leader_acceleration = 0
        self.cur_leader_cumulative_speed = 0

        self.game_object_list.append(self.leader)
        self.game_object_list.append(self.follower)

    def _pos_follower_behind_leader(self):
        follower_start_distance_from_leader = random.randrange(int(self.min_distance * 1.1),
                                                               int(self.max_distance * 0.9), 1)
        follower_start_position_theta = angle_correction(self.leader.direction + 180)

        follower_start_position = np.array(
            (follower_start_distance_from_leader * cos(radians(follower_start_position_theta)),
             follower_start_distance_from_leader * sin(radians(follower_start_position_theta)))) + self.leader.position

        follower_direction = angle_to_point(follower_start_position, self.leader.position)

        self.follower.position = follower_start_position
        self.follower.direction = follower_direction
        self.follower.start_direction = follower_direction

    def _create_obstacles(self):

        #####################################
        # TODO: отсутствие абсолютных чисел!
        self.most_point1 = (self.DISPLAY_WIDTH / 2, 230)
        self.most_point2 = (self.DISPLAY_WIDTH / 2, 770)

        # верхняя и нижняя часть моста
        self.obstacles1 = GameObject('wall',
                                     image=self.wall_img,
                                     start_position=self.most_point1,
                                     height=460,
                                     width=40)

        self.obstacles2 = GameObject('wall',
                                     image=self.wall_img,
                                     start_position=self.most_point2,
                                     height=460,
                                     width=40)

        self.bridge_point = np.array(((self.most_point1[0] + self.most_point2[0]) / 2,
                                      (self.most_point1[1] + self.most_point2[1]) / 2), dtype=np.float32)

        ####################################
        self.obstacles = list()

        wall_start_x = self.obstacles1.rectangle.left
        wall_end_x = self.obstacles1.rectangle.right

        obstacle_size = 50
        bridge_rectangle = pygame.Rect(wall_start_x - self.leader.width * 3,
                                       self.obstacles1.rectangle.bottom - self.leader.height,
                                       self.obstacles1.rectangle.width + 6 * self.leader.width,
                                       self.obstacles2.rectangle.top - self.obstacles1.rectangle.bottom + 2 * self.leader.height)
        for i in range(self.obstacle_number):

            is_free = False

            while not is_free:
                generated_position = (random.randrange(130, self.DISPLAY_WIDTH - 120, self.step_grid),
                                      random.randrange(20, self.DISPLAY_HEIGHT - 20, self.step_grid))

                if self.leader.rectangle.collidepoint(generated_position) or \
                        self.follower.rectangle.collidepoint(generated_position) or \
                        ((generated_position[0] >= wall_start_x) and (generated_position[0] <= wall_end_x)) or \
                        bridge_rectangle.collidepoint(generated_position) or \
                        (distance.euclidean(self.leader.position, generated_position) <= (
                                self.max_distance) + obstacle_size / 2):
                    # чтобы вокруг лидера на минимальном расстоянии не было препятствий (чтобы спокойно генерировать
                    # ведомого за ним)
                    is_free = False
                else:
                    is_free = True

            self.obstacles.append(GameObject('rock',
                                             image=self.rock_img,
                                             start_position=generated_position,
                                             height=obstacle_size,
                                             width=obstacle_size))

        self.game_object_list.append(self.obstacles1)
        self.game_object_list.append(self.obstacles2)
        self.game_object_list.extend(self.obstacles)

    def _init_reward_flags(self):
        self.stop_signal = False
        self.is_in_box = False
        self.is_on_trace = False
        self.follower_too_close = False
        self.crash = False


    def _create_dyn_obs(self):

        self.bears_obs = list()
        bear_size = 25
        bear_speed_coeff = 1.1

        for i in range(self.bear_number):
            # TODO:
            koeff = 90*(i+1)
            if self.bear_behind:
                # bear_start_position = (self.leader.position[0] + koeff, self.leader.position[1] - koeff)
                bear_start_position = (10, 10)

            else:
                bear_start_position = (self.leader.position[0] - koeff, self.leader.position[1] - koeff)

            self.game_dynamic_list.append(AbstractRobot("bear",
                                      image=self.bear_img,
                                      height=bear_size,
                                      width=bear_size,
                                      min_speed=self.leader_config["min_speed"],
                                      max_speed=bear_speed_coeff * self.leader_config["max_speed"],
                                      max_speed_change=self._to_pixels(0.005),  # / 100,
                                      max_rotation_speed=self.leader_config["max_rotation_speed"],
                                      max_rotation_speed_change=20 / 100,
                                      start_position=bear_start_position,
                                    ))


        self.cur_points_for_bear = [bear_start_position] * self.bear_number
        self.dynamics_index = [0]*self.bear_number
        self.dynamics_index_lead = [0] * self.bear_number
        return 0

    def _reset_pose_bear(self):
        for i in range(len(self.game_dynamic_list)):
            koeff = 150 * (i + 1)
            bear_start_position = (self.leader.position[0] + koeff, self.leader.position[1] - koeff)
            self.game_dynamic_list[i].position = bear_start_position

    def _pos_bears_nearest_leader(self):

        for index in range(len(self.game_dynamic_list)):

            ramdom_koeff = 150
            if index==0 or switch:
                switch = False
                bear_start_position = (self.leader.position + rotateVector(np.array([ramdom_koeff, 0]),
                                                                           self.leader.direction - 160))
                bear_direction = self.leader.direction - 90

            else:
                switch = True
                bear_start_position = (self.leader.position + rotateVector(np.array([ramdom_koeff, 0]),
                                                                           self.leader.direction + 160))
                bear_direction = self.leader.direction + 90


            self.game_dynamic_list[index].position = bear_start_position
            self.game_dynamic_list[index].direction = bear_direction
            self.game_dynamic_list[index].start_direction = bear_direction

        return 0

    def _choose_point_around_lid(self, index):

        if distance.euclidean(self.game_dynamic_list[index].position, self.cur_points_for_bear[index]) < self.leader_pos_epsilon:
            self.dynamics_index[index] += 1
            if self.dynamics_index[index] > 3:
                self.dynamics_index[index] = 0

        koeff = 90*(index+1)

        p1 = (self.leader.position[0] + koeff, self.leader.position[1] + koeff)
        p2 = (self.leader.position[0] - koeff, self.leader.position[1] + koeff)
        p3 = (self.leader.position[0] - koeff, self.leader.position[1] - koeff)
        p4 = (self.leader.position[0] + koeff, self.leader.position[1] - koeff)

        if index == 0 :
            dyn_points_list = [p3, p4, p1, p2]
        else:
            dyn_points_list = [p3, p2, p1, p4]

        # dyn_points_list = [p3, p4, p1, p2]

        cur_point = dyn_points_list[self.dynamics_index[index]]

        return cur_point
    def _choose_points_for_bear_stat(self, index):
        if distance.euclidean(self.game_dynamic_list[index].position, self.cur_points_for_bear[index]) < self.leader_pos_epsilon:
            self.dynamics_index[index] += 1
            if self.dynamics_index[index] > 1:
                self.dynamics_index[index] = 0

        bears_points_behind_leader = []
        # ramdom_koeff = random.randrange(int(100), int(300), 10)
        ramdom_koeff = 150*(index+1)
        # print(ramdom_koeff)
        if index >= 0:
            p1 = (self.leader.position + rotateVector(np.array([ramdom_koeff, 0]),
                                                         self.leader.direction - 140))
            p2 = (self.leader.position + rotateVector(np.array([ramdom_koeff, 0]),
                                                         self.leader.direction + 140))

            dyn_points_list = [p1, p2]
            cur_point = dyn_points_list[self.dynamics_index[index]]

        return cur_point


    def _choose_move_bears_points(self, index):

        if distance.euclidean(self.game_dynamic_list[index].position, self.cur_points_for_bear[index]) < self.leader_pos_epsilon:
            self.dynamics_index[index] += 1
            if self.dynamics_index[index] > 3:
                self.dynamics_index[index] = 0

        if index == 0:
            koeff = 70
            p1 = (self.follower.position[0] + koeff, self.follower.position[1] + koeff)
            p2 = (self.follower.position[0] - koeff, self.follower.position[1] + koeff)
            p3 = (self.follower.position[0] - koeff, self.follower.position[1] - koeff)
            p4 = (self.follower.position[0] + koeff, self.follower.position[1] - koeff)
            dyn_points_list = [p1, p2, p3, p4]
            cur_point = dyn_points_list[self.dynamics_index[index]]

        elif index == 1:
            koeff = 100
            p1 = (self.follower.position[0] + koeff, self.follower.position[1] + koeff)
            p2 = (self.follower.position[0] - koeff, self.follower.position[1] + koeff)
            p3 = (self.follower.position[0] - koeff, self.follower.position[1] - koeff)
            p4 = (self.follower.position[0] + koeff, self.follower.position[1] - koeff)
            dyn_points_list = [p4, p3, p2, p1]
            cur_point = dyn_points_list[self.dynamics_index[index]]

        elif index == 2:

            koeff = 150

            p1 = (self.follower.position[0] + koeff, self.follower.position[1] + koeff)
            p2 = (self.follower.position[0] - koeff, self.follower.position[1] + koeff)
            p3 = (self.follower.position[0] - koeff, self.follower.position[1] - koeff)
            p4 = (self.follower.position[0] + koeff, self.follower.position[1] - koeff)

            dyn_points_list = [p1, p2, p3, p4]
            cur_point = dyn_points_list[self.dynamics_index[index]]

        else:
            dyn_points_list = [0, 0, 0, 0]
            for i in range(4):
                dyn_points_list[i] = (random.randrange(self.max_distance, self.DISPLAY_WIDTH - self.max_distance, 10),
                                      random.randrange(self.max_distance, self.DISPLAY_HEIGHT - self.max_distance, 10))

            cur_point = dyn_points_list[self.dynamics_index[index]]

        return cur_point

    def _chose_cur_point_for_leader(self, dyn_obstacle_pose, cur_dyn_point):

        koeff = 150

        p1 = (self.leader.position[0] + koeff, self.leader.position[1] + koeff)
        p2 = (self.leader.position[0] - koeff, self.leader.position[1] + koeff)
        p3 = (self.leader.position[0] - koeff, self.leader.position[1] - koeff)
        p4 = (self.leader.position[0] + koeff, self.leader.position[1] - koeff)

        dyn_points_list = [p1, p2, p3, p4]
        min_dist = 5000
        # for i in range(len(dyn_points_list)):
        for i in dyn_points_list:
            dist = distance.euclidean(dyn_obstacle_pose, i)
            if dist <= min_dist:
                min_dist=dist
                cur_point = i

        return cur_point

    def step(self, action):
        # Если контролирует автомат, то нужно преобразовать угловую скорость с учётом её знака.
        if self.constant_follower_speed:
            self.follower.command_forward(self.follower.max_speed + self.PIXELS_TO_METER)

        if self.manual_control:
            for event in pygame.event.get():
                self.manual_game_contol(event, self.follower)

        else:
            if self.discrete_action_space:
                if type(action) is np.ndarray:
                    assert action.shape[0] == 1 and action.shape[1] == 1
                    action = action[0, 0]
                action = (self.follower.max_speed, self.discrete_rotation_speed_to_value[action])

            if self.constant_follower_speed:
                action = np.concatenate([[0.25], action])

            self.follower.command_forward(action[0])
            if action[1] < 0:
                self.follower.command_turn(abs(action[1]), -1)
            elif action[1] > 0:
                self.follower.command_turn(action[1], 1)
            else:
                self.follower.command_turn(0, 0)

        for cur_ministep_nb in range(self.frames_per_step):
            obs, reward, done, info = self.frame_step(action)
        self.follower_scan_dict = self.follower.use_sensors(self)
        obs = self._get_obs()
        if self.random_frames_per_step is not None:
            self.frames_per_step = np.random.randint(self.random_frames_per_step[0], self.random_frames_per_step[1])
        return obs, reward, done, info

    def frame_step(self, action):
        """Стандартный для gym обработчик одного шага среды (в данном случае один кадр)"""
        self.is_in_box = False
        self.is_on_trace = False
        info = {
            "mission_status": "in_progress",
            "agent_status": "moving",
            "leader_status": "moving"
        }

        self.follower.move()

        # определение столкновения ведомого с препятствиями
        if self._collision_check(self.follower):
            self.crash = True
            self.done = True
            info["mission_status"] = "fail"
            info["agent_status"] = "crash"

        # Определение коробки и агента в ней
        # определение текущих точек маршрута, которые являются подходящими для Агента
        self.green_zone_trajectory_points = list()
        self._trajectory_in_box()
        # self._get_green_zone_border_points()

        # определяем положение Агента относительно маршрута и коробки
        self._check_agent_position()

        # работа с движением лидера
        prev_leader_position = self.leader.position.copy()

        if distance.euclidean(self.leader.position, self.cur_target_point) < self.leader_pos_epsilon:
            self.cur_target_id += 1
            if self.cur_target_id >= len(self.trajectory):
                self.leader_finished = True
            else:
                self.cur_target_point = self.trajectory[self.cur_target_id]

        # TODO : Добавить движение динамичкеского препятствия тут №№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№3

        if self.add_bear:

            for cur_dyn_obj_index in range(0, len(self.game_dynamic_list)):

                if self.multi_random_bears:

                    if distance.euclidean(self.leader.position, self.game_dynamic_list[cur_dyn_obj_index].position) < 150:
                        # print("ALARM")
                        self.cur_points_for_bear[cur_dyn_obj_index] = self._chose_cur_point_for_leader(
                            self.game_dynamic_list[cur_dyn_obj_index].position, 1)
                        self.game_dynamic_list[cur_dyn_obj_index].move_to_the_point(
                            self.cur_points_for_bear[cur_dyn_obj_index])

                    else:
                        self.cur_points_for_bear[cur_dyn_obj_index] = self._choose_move_bears_points(cur_dyn_obj_index)
                        self.game_dynamic_list[cur_dyn_obj_index].move_to_the_point(self.cur_points_for_bear[cur_dyn_obj_index])

                else:
                    # TODO : debug, может стоит поправить в будущем

                    if self.bear_behind:
                        # TODO : test 1
                        self.cur_points_for_bear[cur_dyn_obj_index] = self._choose_points_for_bear_stat(cur_dyn_obj_index)
                    else:
                        # TODO : test 2
                        self.cur_points_for_bear[cur_dyn_obj_index] = self._choose_point_around_lid(cur_dyn_obj_index)

                    self.game_dynamic_list[cur_dyn_obj_index].move_to_the_point(self.cur_points_for_bear[cur_dyn_obj_index])

                    # TODO : test 3
                    # if distance.euclidean(self.leader.position,
                    #                       self.game_dynamic_list[cur_dyn_obj_index].position) < 80:
                    #     self.cur_points_for_bear[cur_dyn_obj_index] = self._choose_point_around_lid(cur_dyn_obj_index)
                    #     self.game_dynamic_list[cur_dyn_obj_index].move_to_the_point(
                    #         self.cur_points_for_bear[cur_dyn_obj_index], speed=0)
                    # else:
                    #     self.cur_points_for_bear[cur_dyn_obj_index] = self._choose_point_around_lid(cur_dyn_obj_index)
                    #     self.game_dynamic_list[cur_dyn_obj_index].move_to_the_point(
                    #         self.cur_points_for_bear[cur_dyn_obj_index])



        # TODO : Добавить движение динамичкеского препятствия тут №№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№3

        if not self.leader_finished:
            if self.leader_speed_regime is not None:
                speed = self._process_leader_speed_regime()
            else:
                speed = self.leader.max_speed

            if self.leader_acceleration_regime is not None:
                acceleration = self._process_leader_acceleration_regime() / self.frames_per_step
            else:
                acceleration = 0
            self.leader.move_to_the_point(self.cur_target_point, speed=speed + acceleration)



        else:
            self.leader.command_forward(0)
            self.leader.command_turn(0, 0)
            info["leader_status"] = "finished"

        # обработка столкновений лидера
        if self._collision_check(self.leader):
            print("Лидер столкнулся с препятствием!")
            self.done = True
            info["mission_status"] = "fail"
            info["leader_status"] = "crash"

        if pygame.time.get_ticks() % self.trajectory_saving_period == 0:
            self.leader_factual_trajectory.append(self.leader.position.copy())

        if self.leader_finished and self.is_in_box:
            if self.finish_position_framestimer is None:
                self.finish_position_framestimer = 0
            else:
                self.finish_position_framestimer += 1
                if self.finish_position_framestimer > self.frames_per_step * 20:
                    info["mission_status"] = "success"
                    info["leader_status"] = "finished"
                    info["agent_status"] = "finished"
                    # TODO : вернуть
                    self.done = True
        if self.step_count > self.warm_start:
            if "low_reward" in self.early_stopping and self.accumulated_penalty < self.early_stopping["low_reward"]:
                # print("LOW REWARD")
                info["mission_status"] = "fail"
                info["leader_status"] = "moving"
                info["agent_status"] = "low_reward"
                # TODO : вернуть
                self.crash = True
                self.done = True

            if "max_distance_coef" in self.early_stopping and np.linalg.norm(
                    self.follower.position - self.leader.position) > self.max_distance * self.early_stopping[
                "max_distance_coef"]:
                # print("FOLLOWER IS TOO FAR")
                info["mission_status"] = "fail"
                info["leader_status"] = "moving"
                info["agent_status"] = "too_far_from_leader"
                # TODO : вернуть
                self.crash = True
                self.done = True

        res_reward = self._reward_computation()
        if res_reward < 0:
            self.accumulated_penalty += res_reward
        else:
            self.accumulated_penalty = 0

        self.overall_reward += res_reward

        self.clock.tick(self.framerate)

        if self.simulation_time_limit is not None:
            if pygame.time.get_ticks() * 1000 > self.simulation_time_limit:
                # TODO : вернуть
                self.done = True
                print("Время истекло! Прошло {} секунд.".format(self.simulation_time_limit))

        obs = self._get_obs()

        self.step_count += 1

        if self.step_count > self.max_steps:
            info["mission_status"] = "finished_by_time"
            info["leader_status"] = "moving"
            info["agent_status"] = "moving"
            # TODO : вернуть
            self.done = True

        if self.aggregate_reward:
            reward_to_return = self.overall_reward
        else:
            reward_to_return = res_reward

        return obs, reward_to_return, self.done, info

    def _process_leader_speed_regime(self):
        """Функция обрабатывает словарь скорости движения лидера."""
        min_step_distance = np.inf

        for cur_key in list(self.leader_speed_regime.keys()):
            if cur_key <= self.step_count:
                if abs(self.step_count - cur_key) < min_step_distance:
                    self.cur_speed_multiplier = self.leader_speed_regime[cur_key]
            elif cur_key > self.step_count:
                continue
                #del self.leader_speed_regime[cur_key]

        if type(self.cur_speed_multiplier) in (tuple, list):
            self.cur_speed_multiplier = random.uniform(self.cur_speed_multiplier[0], self.cur_speed_multiplier[1])
        return self.leader.max_speed * self.cur_speed_multiplier

    def _process_leader_acceleration_regime(self):
        """Функция обрабатывает словарь ускорения движения лидера."""
        min_step_distance = np.inf

        acceleration = 0
        for cur_key in list(self.leader_acceleration_regime.keys()):
            if cur_key <= self.step_count:
                if abs(self.step_count - cur_key) < min_step_distance:
                    self.cur_leader_acceleration = self.leader_acceleration_regime[cur_key]
                    self.cur_leader_cumulative_speed = self.cur_leader_acceleration

                del self.leader_acceleration_regime[cur_key]

        self.cur_leader_cumulative_speed += self.cur_leader_acceleration

        return self.cur_leader_cumulative_speed * self.leader.max_speed

    def _collision_check(self, target_object):
        """Рассматривает, не участвует ли объект в коллизиях"""
        objects_to_collide = [cur_obj.rectangle for cur_obj in self.game_object_list if cur_obj is not target_object]
        dyn_objects_to_collide = [cur_obj.rectangle for cur_obj in self.game_dynamic_list if cur_obj is not target_object]

        if (target_object.rectangle.collidelist(objects_to_collide) != -1) or \
                (target_object.rectangle.collidelist(dyn_objects_to_collide) != -1) or \
                any(target_object.position > (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)) or \
                any(target_object.position < 0):
            return True
        else:
            return False

    def render(self, custom_message=None, **kwargs):
        """Стандартный для gym метод отображения окна и обработки событий в нём (например, нажатий клавиш)"""

        self._show_tick()
        pygame.display.update()
        return np.transpose(
            pygame.surfarray.array3d(self.gameDisplay), axes=(1, 0, 2))

    def rotate_object(self, object_to_rotate):
        """Поворачивает изображение объекта, при этом сохраняя его центр и прямоугольник для взаимодействия.
        """
        cur_rect = object_to_rotate.rectangle
        # Rotate the original image without modifying it.
        new_image = pygame.transform.rotate(object_to_rotate.image, -object_to_rotate.direction)
        # Get a new rect with the center of the old rect.
        # Изменение хитбокса убрал в самих роботов
        # object_to_rotate.rectangle = new_image.get_rect(center=cur_rect.center)

        return new_image

    def show_object(self, object_to_show):
        """Отображает объект с учётом его направления"""
        cur_image = object_to_show.image
        if hasattr(object_to_show, "direction"):
            cur_image = self.rotate_object(object_to_show)
        self.gameDisplay.blit(cur_image, (
            object_to_show.position[0] - object_to_show.width / 2,
            object_to_show.position[1] - object_to_show.height / 2))

        if self.show_rectangles:
            pygame.draw.rect(self.gameDisplay, self.colours["red"], object_to_show.rectangle, width=1)

    def _show_tick(self):
        """Отображает всё, что положено отображать на каждом шаге"""
        self.gameDisplay.fill(self.colours["white"])  # фон
        if self.add_obstacles:
            pygame.draw.circle(self.gameDisplay, self.colours["black"], self.bridge_point, 5)
        pygame.draw.circle(self.gameDisplay, self.colours["red"], self.finish_point, 5)
        if self.multiple_end_points:
            pygame.draw.circle(self.gameDisplay, self.colours["red"], self.finish_point2, 5)
            pygame.draw.circle(self.gameDisplay, self.colours["red"], self.finish_point3, 5)
        # отображение полного маршрута Ведущего
        if self.show_leader_path:
            if len(self.trajectory) > 2:
                pygame.draw.aalines(self.gameDisplay, self.colours["red"], False, self.trajectory)

        # отображение зоны, в которой нужно находиться Ведомому
        if self.show_box:
            if len(self.green_zone_trajectory_points) > 5:
                #                 green_line = pygame.draw.polygon(self.gameDisplay,self.colours["green"],self.green_zone_trajectory_points[::5], width=self.max_dev*2)
                green_line = pygame.draw.lines(self.gameDisplay,
                                               self.colours["green"],
                                               False,
                                               self.green_zone_trajectory_points[::4],
                                               width=int(self.max_dev * 2))

                # for cur_point in self.left_border_points_list:
                #    pygame.draw.circle(self.gameDisplay, self.colours["black"], cur_point, 1)

                # for cur_point in self.right_border_points_list:
                #    pygame.draw.circle(self.gameDisplay, self.colours["black"], cur_point, 1)

        # отображение пройденной Ведущим траектории
        if self.show_leader_trajectory:
            for cur_point in self.leader_factual_trajectory[::10]:  # Каждую 10ю точку показываем.
                pass
                # pygame.draw.circle(self.gameDisplay, self.colours["black"], cur_point, 3)

        # отображение всех игровых объектов, которые были добавлены в список игровых объектов
        for cur_object in self.game_object_list:
            self.show_object(cur_object)

        for cur_dyn_object in self.game_dynamic_list:
            self.show_object(cur_dyn_object)

        # TODO: здесь будет отображение препятствий (лучше, если в рамках цикла выше, то есть как игровых объектов)
        #  [Слава]
        # отображение круга минимального расстояния
        if self.follower_too_close:
            close_circle_width = 2
        else:
            close_circle_width = 1

        self.leader_close_circle = pygame.draw.circle(self.gameDisplay, self.colours["red"], self.leader.position,
                                                      self.min_distance, width=close_circle_width)

        if self.show_sensors:
            for sensor_name, cur_sensor in self.follower.sensors.items():
                cur_sensor.show(self)

                pygame.draw.circle(self.gameDisplay, self.colours["red"], self.cur_target_point, 10, width=2)
        #         if self.add_obstacles:
        #             pygame.draw.circle(self.gameDisplay, self.colours["black"], self.first_bridge_point, 10, width=3)
        #             pygame.draw.circle(self.gameDisplay, self.colours["black"], self.second_bridge_point, 10, width=3)
        reward_text = self.font.render("Step: {}, Суммарная награда:{}".format(self.step_count,
                                                                               self.overall_reward),
                                       False,
                                       (0, 0, 0))
        self.gameDisplay.blit(reward_text, (0, 0))
        reward_text = self.font.render("Фолловер. скорость:{}, скорость поворота:{}".format(self.follower.speed,
                                                                                            self.follower.rotation_speed),
                                       False,
                                       (0, 0, 0))
        self.gameDisplay.blit(reward_text, (0, 50))
        reward_text = self.font.render("Лидер. скорость:{}, скорость поворота:{}".format(self.leader.speed,
                                                                                         self.leader.rotation_speed),
                                       False,
                                       (0, 0, 0))
        self.gameDisplay.blit(reward_text, (0, 100))

    def generate_trajectory(self, n=8, min_distance=30, border=20, parent=None, position=None, iter_limit=10000):
        """Случайно генерирует точки на карте, по которым должен пройти ведущий"""
        # TODO: добавить проверку, при которойо точки не на одной прямой
        # TODO: добавить отдельную функцию, которая использует эту:
        # на вход принимает шаблон -- список из r и c, где
        #    r -- placeholder, на место которого будут подставляться случайные точки
        #    c -- координаты точки, которые точно должны присутствовать в пути (например, координаты "моста")
        # TODO: вообще нужен отдельный класс для траекторий;
        # TODO: если строить маршрут с учётом препятствий сразу, вероятно обработка будет здесь или где-то рядом [Слава]
        # TODO: ограничение на число итераций цикла (иначе может уйти в бесконечность).
        # вероятно нужно сделать staticmethod
        #  генерация финишной точки
        # self.finish_point = np.float64((random.randrange(20, 500,10),random.randrange(20, 500,10)))

        # шаг сетки для вычислений. Если менять коэф, то надо изменить и в atar file в def return_path
        # self.step_grid
        # step_grid = 10
        t_astar = time.time()

        step_obs = 60 / self.step_grid

        print(self.leader.start_position)
        print(self.finish_point)

        self.wid = self.DISPLAY_WIDTH
        self.hit = self.DISPLAY_HEIGHT

        start = (
            int(self.leader.start_position[0] / self.step_grid), int(self.leader.start_position[1] / self.step_grid))
        # int(start)
        # start.tolist(start)
        print(start)
        end = (int(self.finish_point[0] / self.step_grid), int(self.finish_point[1] / self.step_grid))
        # int(end)
        print(end)

        wid = int(self.wid / self.step_grid)
        print(wid)
        hit = int(self.hit / self.step_grid)
        print(hit)

        # print(self.start)
        # print(self.end)

        grid = []
        for i in range(wid):
            grid.append([0] * hit)

        for i in range(wid):
            for j in range(hit):
                for k in range(self.obstacle_number):
                    ob = (self.obstacles[k].start_position / self.step_grid)
                    ob = ob.astype(int)
                    # print(ob)
                    if distance.euclidean((i, j), ob) < step_obs:
                        grid[i][j] = 1
                    if i >= 700 / self.step_grid and i <= 800 / self.step_grid and j >= 0 and j <= 480 / self.step_grid:
                        grid[i][j] = 1
                    if i >= 700 / self.step_grid and i <= 800 / self.step_grid and j >= 530 / self.step_grid \
                            and j <= 1000 / self.step_grid:
                        grid[i][j] = 1

        # print(grid)
        path = astar(maze=grid, start=start, end=end)
        # print(path)
        #         print(grid)
        trajectory = []
        trajectory = path
        timeAstar = time.time() - t_astar

        # # print(trajectory)
        # # print(self.obstacles)
        # len_tr = len(trajectory)
        # # print('Astr: ', len_tr)
        # # print('Aend =', timeAstar)
        # path_tab_astar = '~/Desktop/lentrAstar.xlsx'
        # astar_table = pd.read_excel(path_tab_astar, index_col=False)
        # # print(astar_table)
        # # print(len_tr)
        # # print(timeAstar)
        # new_data = {'Time':timeAstar, 'Len':len_tr}
        # new_astar_table = astar_table.append(new_data, ignore_index=True)
        # # print(new_astar_table)
        # #astar_table.loc[len(astar_table.index)] = ['timeAstar', 'len_tr']
        # # pdc = pd.DataFrame(new_astar_table, index=False)
        # new_astar_table.to_excel(path_tab_astar, index=False)

        return trajectory

    # Алгоритм поиска RRT
    def generate_trajectory_rrt(self):

        obstacle_list = []  # [x,y,size(radius)]

        print(self.leader.start_position)
        print(self.finish_point)

        for i in range(self.obstacle_number):
            obst = (self.obstacles[i].start_position[0] / self.step_grid,
                    self.obstacles[i].start_position[1] / self.step_grid,
                    (100 / self.step_grid) / 2)
            obstacle_list.append(obst)
        print(obstacle_list)

        t_rrt = time.time()

        # Set Initial parameters
        # rrt = RRT(
        #     start=self.leader.start_position/self.step_grid,
        #     goal= ((self.finish_point[0])/self.step_grid,
        #            (self.finish_point[1])/self.step_grid), #self.finish_point,
        #     rand_area=[0, 110],
        #     expand_dis=1.5,
        #     path_resolution=2,
        #     goal_sample_rate=1,
        #     max_iter=1500,
        #     obstacle_list=obstacle_list)

        rrt = RRT(
            start=self.leader.start_position / self.step_grid,
            goal=((self.finish_point[0]) / self.step_grid,
                  (self.finish_point[1]) / self.step_grid),  # self.finish_point,
            rand_area=[0, 110],
            expand_dis=2.5,
            path_resolution=2,
            goal_sample_rate=1,
            max_iter=1500,
            obstacle_list=obstacle_list)

        path = rrt.planning(animation=False)
        # print(path)
        # trajectory = rrt_star.planning(animation=False)

        trajectory = []
        trajectory = path[::-1]
        trajectory.pop(0)

        time_rrt = time.time() - t_rrt
        len_rrt = len(trajectory)
        # print('Dstr: ', len_rrt)
        # print('Dend =', time_rrt)
        path_tab_rrt = '~/Desktop/lentRRT.xlsx'
        rrt_table = pd.read_excel(path_tab_rrt, index_col=False)
        # print(rrtstar_table)
        # print(len_rrtstar)
        # print(time_rrtstar)
        new_data = {'Time': time_rrt, 'Len': len_rrt}
        new_dstar_table = rrt_table.append(new_data, ignore_index=True)
        new_dstar_table.to_excel(path_tab_rrt, index=False)

        #         print(trajectory)
        # print(grid[75][23])

        return trajectory

    # Алгоритм поиска RRTstar
    def generate_trajectory_rrtstar(self):

        obstacle_list = []  # [x,y,size(radius)]

        # print(self.leader.start_position)
        # print(self.finish_point)

        for i in range(self.obstacle_number):
            obst = (self.obstacles[i].start_position[0] / self.step_grid,
                    self.obstacles[i].start_position[1] / self.step_grid,
                    (80 / self.step_grid) / 2)
            obstacle_list.append(obst)
        # print(obstacle_list)

        for k in range(20, 460, 40):
            most1 = (750 / self.step_grid, k / self.step_grid, 20 / self.step_grid)
            obstacle_list.append(most1)

        for k in range(560, 1000, 40):
            most2 = (750 / self.step_grid, k / self.step_grid, 20 / self.step_grid)
            obstacle_list.append(most2)

        # print(obstacle_list)
        t_rrtstar = time.time()
        # Set Initial parameters
        rrt_star = RRTStar(
            start=self.leader.start_position / self.step_grid,
            goal=self.finish_point / self.step_grid,
            # (50,50),#((self.leader.start_position[0]+200)/self.step_grid,
            # (self.leader.start_position[1]-200)/self.step_grid),
            rand_area=[0, 150],
            obstacle_list=obstacle_list,
            expand_dis=20, goal_sample_rate=20, path_resolution=1, connect_circle_dist=50)

        path = rrt_star.planning(animation=False)

        trajectory = []
        trajectory = path[::-1]
        trajectory.pop(0)
        # print(trajectory)
        # print(grid[75][23])

        time_rrtstar = time.time() - t_rrtstar
        len_rrtstar = len(trajectory)
        # print('Dstr: ', len_rrtstar)
        # print('Dend =', time_rrtstar)
        path_tab_rrtstar = '~/Desktop/lentRRTstar.xlsx'
        rrtstar_table = pd.read_excel(path_tab_rrtstar, index_col=False)
        # print(rrtstar_table)
        # print(len_rrtstar)
        # print(time_rrtstar)
        new_data = {'Time': time_rrtstar, 'Len': len_rrtstar}
        new_dstar_table = rrtstar_table.append(new_data, ignore_index=True)
        new_dstar_table.to_excel(path_tab_rrtstar, index=False)

        return trajectory

    # Алгоритм поиска LQR RRTstar
    def generate_trajectory_lqr_rrtstar(self):

        obstacle_list = []  # [x,y,size(radius)]

        # print(self.leader.start_position)
        # print(self.finish_point)

        for i in range(self.obstacle_number):
            obst = (self.obstacles[i].start_position[0] / self.step_grid,
                    self.obstacles[i].start_position[1] / self.step_grid,
                    (80 / self.step_grid) / 2)
            obstacle_list.append(obst)
        # print(obstacle_list)

        for k in range(20, 460, 40):
            most1 = (750 / self.step_grid, k / self.step_grid, 20 / self.step_grid)
            obstacle_list.append(most1)

        for k in range(560, 1000, 40):
            most2 = (750 / self.step_grid, k / self.step_grid, 20 / self.step_grid)
            obstacle_list.append(most2)

        # print(obstacle_list)

        lqr_rrt_star = LQRRRTStar(self.leader.start_position / self.step_grid, (90, 90),
                                  # self.finish_point/self.step_grid,
                                  obstacle_list,
                                  [0, 100.0])
        path = lqr_rrt_star.planning(animation=False)

        trajectory = []
        trajectory = path[::-1]
        trajectory.pop(0)
        # print(trajectory)

        return trajectory

    # Алгоритм поиска Dstar (еще не настроен)
    def generate_trajectory_dstar(self):

        # t_dstar = time.time()
        # print('D0 =', time.time()-t_astar)
        # print(self.leader.start_position)
        # print(self.finish_point)
        # print(self.finish_point2)
        # print(self.finish_point3)

        m = Map(150, 100)
        m2 = Map(150, 100)
        m3 = Map(150, 100)

        ox, oy = [], []

        for ob in range(self.obstacle_number):
            for i in range(-50, 50, 10):
                for j in range(-50, 50, 10):
                    ox.append(int((self.obstacles[ob].start_position[0] + i) / self.step_grid))
                    oy.append(int((self.obstacles[ob].start_position[1] + j) / self.step_grid))

        for k in range(0, 490, 10):
            for i in range(-40, 40, 10):
                # for j in range(-30,30,10):
                ox.append(int((750 + i) / self.step_grid))
                oy.append(int((k) / self.step_grid))

        for k in range(520, 1000, 10):
            for i in range(-40, 40, 10):
                # for j in range(-30,30,10):
                ox.append(int((750 + i) / self.step_grid))
                oy.append(int((k) / self.step_grid))

        # print([(i, j) for i, j in zip(ox, oy)])
        m.set_obstacle([(i, j) for i, j in zip(ox, oy)])

        m2.set_obstacle([(i, j) for i, j in zip(ox, oy)])
        m3.set_obstacle([(i, j) for i, j in zip(ox, oy)])

        ########### работа dstar 1
        start = [int(self.leader.start_position[0] / self.step_grid),
                 int(self.leader.start_position[1] / self.step_grid)]
        goal = [int(self.finish_point[0] / self.step_grid),
                int(self.finish_point[1] / self.step_grid)]

        start = m.map[start[0]][start[1]]
        end = m.map[goal[0]][goal[1]]
        dstar = Dstar(m)
        rx, ry, found_target_point = dstar.run(start, end)
        trajectory = []
        # trajectory = path[::-1]
        for i in range(len(rx)):
            trajectory.append((rx[i], ry[i]))
        # print(trajectory)

        if self.multiple_end_points:
            # TODO : работа дстар в 3 захода (костыль)
            # ##############################################################
            # работа dstar 2
            # start2 = [int(self.leader.start_position[0]/self.step_grid),
            #         int(self.leader.start_position[1]/self.step_grid)]

            start2 = goal
            goal2 = [int(self.finish_point2[0] / self.step_grid),
                     int(self.finish_point2[1] / self.step_grid)]

            start2 = m2.map[start2[0]][start2[1]]
            end2 = m2.map[goal2[0]][goal2[1]]
            dstar2 = Dstar(m2)
            rx2, ry2, found_target_point = dstar2.run(start2, end2)
            trajectory2 = []
            # trajectory = path[::-1]
            for i in range(len(rx2)):
                trajectory2.append((rx2[i], ry2[i]))
            print(trajectory2)

            # работа dstar 3
            # start3 = [int(self.leader.start_position[0]/self.step_grid),
            #          int(self.leader.start_position[1]/self.step_grid)]

            start3 = goal2
            goal3 = [int(self.finish_point3[0] / self.step_grid),
                     int(self.finish_point3[1] / self.step_grid)]

            start3 = m3.map[start3[0]][start3[1]]
            end3 = m3.map[goal3[0]][goal3[1]]
            dstar3 = Dstar(m3)
            rx3, ry3, found_target_point = dstar3.run(start3, end3)
            trajectory3 = []
            # trajectory = path[::-1]
            for i in range(len(rx3)):
                trajectory3.append((rx3[i], ry3[i]))
            print(trajectory3)
            trajectory = trajectory + trajectory2 + trajectory3

            # print(trajectory)
            # len_dstar = len(trajectory)
            # print("Dstar: ", len_dstar)
            # print('Dend =', time.time() - t_dstar)

            #############################
            # time_dstar = time.time() - t_dstar
            # len_dstar = len(trajectory)
            # if len_dstar>200:
            #     # print('Dstr: ', len_dstar)
            #     # print('Dend =', time_dstar)
            #     path_tab_dstar = '~/Desktop/lentrDstar-1.xlsx'
            #     dstar_table = pd.read_excel(path_tab_dstar, index_col=False)
            #     # print(dstar_table)
            #     # print(len_dstar)
            #     # print(time_dstar)
            #     new_data = {'Time50': time_dstar, 'Len50': len_dstar}
            #     # new_data = {'Time15': time_dstar, 'Len15': len_dstar}
            #     new_dstar_table = dstar_table.append(new_data, ignore_index=True)
            #     new_dstar_table.to_excel(path_tab_dstar, index=False)
            # else:
            #     return "PUSTO"
            # # print('Dstr: ', len_dstar)
            # # print('Dend =', time_dstar)
            # path_tab_dstar = '~/Desktop/lentrDstar-1.xlsx'
            # dstar_table = pd.read_excel(path_tab_dstar, index_col=False)
            # # print(dstar_table)
            # # print(len_dstar)
            # # print(time_dstar)
            # # new_data = {'Time': time_dstar, 'Len': len_dstar}
            # new_data = {'Time1': time_dstar, 'Len1': len_dstar}
            # new_dstar_table = dstar_table.append(new_data, ignore_index=True)
            # new_dstar_table.to_excel(path_tab_dstar, index=False)

        return trajectory

    def generate_finish_point(self, left_top_border=[20, 20], right_bottom_border=None):
        if right_bottom_border is None:
            right_bottom_border = [int(self.DISPLAY_HEIGHT / 2), int(self.DISPLAY_WIDTH / 2)]
        #  генерация финишной точки
        correct_point_position = False
        while not correct_point_position:

            correct_point_position = True
            generated_finish_point = (random.randrange(left_top_border[0], right_bottom_border[0], 10),
                                      random.randrange(left_top_border[1], right_bottom_border[1], 10))

            for cur_object in self.game_object_list:
                if (cur_object.rectangle.collidepoint(generated_finish_point)) or \
                        (distance_to_rect(generated_finish_point, cur_object) < self.leader_pos_epsilon):
                    correct_point_position = False

        return generated_finish_point

    def generate_trajectory_astar(self,
                                  max_iter=None):
        """Случайно генерирует точки на карте, по которым должен пройти ведущий, строит маршрут методом A-star"""

        # шаг сетки для вычислений. Если менять коэф, то надо изменить и в atar file в def return_path
        step_grid = 20

        start = (int(self.leader.start_position[0] / step_grid),
                 int(self.leader.start_position[1] / step_grid))

        end = (int(self.finish_point[0] / step_grid),
               int(self.finish_point[1] / step_grid))

        astar_grid_width = int(self.DISPLAY_WIDTH / step_grid)
        astar_grid_height = int(self.DISPLAY_HEIGHT / step_grid)

        grid = np.zeros([astar_grid_width, astar_grid_height], dtype=int)

        leader_size_factor = int((max(self.leader.width, self.leader.height) * 2))

        for cur_obstacle in self.game_object_list:
            if cur_obstacle in {self.leader, self.follower}:
                continue
            else:
                start_x = max(int((cur_obstacle.rectangle.left - leader_size_factor) / step_grid), 0)
                end_x = min(int((cur_obstacle.rectangle.right + leader_size_factor) / step_grid), astar_grid_width - 1)

                start_y = max(int((cur_obstacle.rectangle.top - leader_size_factor) / step_grid), 0)
                end_y = min(int((cur_obstacle.rectangle.bottom + leader_size_factor) / step_grid),
                            astar_grid_height - 1)

                for x_coord in range(start_x, end_x):
                    for y_coord in range(start_y, end_y):
                        grid[x_coord, y_coord] = 1
        if self.add_obstacles:
            bridge_point = np.divide(self.bridge_point, step_grid).astype(int)
            bridge_point = (bridge_point[0], bridge_point[1])
            grid[bridge_point] = 0

            for i in range(int((self.obstacles1.rectangle.left / step_grid) - (leader_size_factor / step_grid)),
                           int((self.obstacles1.rectangle.right / step_grid) + (leader_size_factor / step_grid))):
                grid[i, bridge_point[1]] = 0

            first_bridge_point = (
                int((self.obstacles1.rectangle.right + self.leader_pos_epsilon) / step_grid), bridge_point[1])
            second_bridge_point = (
                int((self.obstacles1.rectangle.left - self.leader_pos_epsilon) / step_grid), bridge_point[1])

            self.first_bridge_point = (
                int((self.obstacles1.rectangle.right + self.leader_pos_epsilon)), step_grid * bridge_point[1])
            self.second_bridge_point = (
                int((self.obstacles1.rectangle.left - self.leader_pos_epsilon)), step_grid * bridge_point[1])

            path = astar(maze=grid,
                         start=start,
                         end=first_bridge_point,
                         max_iterations=max_iter,
                         return_none_on_max_iter=False)

            if path is None:
                return []

            if path[-1] != first_bridge_point:
                path.append(self.first_bridge_point)

            path_continued = astar(maze=grid,
                                   start=second_bridge_point,
                                   end=end,
                                   max_iterations=max_iter,
                                   return_none_on_max_iter=False)
            if path_continued is None:
                return path
            return path + path_continued
        else:
            path = astar(maze=grid,
                         start=start,
                         end=end,
                         max_iterations=max_iter,
                         return_none_on_max_iter=False)
            return path

    def generate_trajectory_old(self, n=8, min_distance=30, border=20, parent=None, position=None, iter_limit=10000):
        """Случайно генерирует точки на карте, по которым должен пройти ведущий"""
        trajectory = list()

        i = 0  # пока отслеживаем зацикливание по числу итераций на генерацию каждой точки. Примитивно, но лучше, чем никак

        while (len(trajectory) < n) and (i < iter_limit):
            new_point = np.array((np.random.randint(border, high=self.DISPLAY_WIDTH - border),
                                  np.random.randint(border, high=self.DISPLAY_HEIGHT - border)))

            if len(trajectory) == 0:
                trajectory.append(new_point)
                i = 0
            else:
                to_add = True

                # работает только на ограниченном числе точек, может уйти в бесконечный цикл, осторожнее!!!

                for prev_point in trajectory:
                    if distance.euclidean(prev_point, new_point) < min_distance:
                        to_add = False

                if to_add:
                    trajectory.append(new_point)

                i += 1
        return trajectory

    def manual_game_contol(self, event, follower):
        """обработчик нажатий клавиш при ручном контроле."""
        # В теории, можно на основе этого класса сделать управляемого руками Ведущего. Но надо модифицировать.

        if event.type == pygame.QUIT:
            self.done = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                if follower.rotation_direction > 0:

                    follower.rotation_speed = 0
                    follower.rotation_direction = 0
                    follower.command_turn(0, 0)
                else:
                    follower.command_turn(follower.rotation_speed + 2, -1)

            if (event.key == pygame.K_RIGHT):
                if follower.rotation_direction < 0:
                    follower.rotation_speed = 0
                    follower.rotation_direction = 0
                    follower.command_turn(0, 0)
                else:
                    follower.command_turn(follower.rotation_speed + 2, 1)

            if not self.constant_follower_speed:
                if event.key == pygame.K_UP:
                    follower.command_forward(follower.speed + self.PIXELS_TO_METER)

                if event.key == pygame.K_DOWN:
                    follower.command_forward(follower.speed - self.PIXELS_TO_METER)

    def _get_obs(self):
        """Возвращает наблюдения (observations) среды каждый шаг (step)"""
        obs_dict = dict()

        obs_dict["numerical_features"] = np.array([self.leader.position[0],
                                                   self.leader.position[1],
                                                   self.leader.speed,
                                                   self.leader.direction,
                                                   self.leader.rotation_speed,
                                                   self.follower.position[0],
                                                   self.follower.position[1],
                                                   self.follower.speed,
                                                   self.follower.direction,
                                                   self.follower.rotation_speed], dtype=np.float32)
        if self.cur_target_point == self.trajectory[-1]:
            obs_dict["leader_target_point"] = self.trajectory[-2]
        else:
            obs_dict["leader_target_point"] = self.cur_target_point

        obs_dict.update(self.follower_scan_dict)

        return obs_dict

    def _create_observation_space(self):
        self.observation_space = Box(low=np.array((0, 0,
                                                   0, 0,
                                                   -self.leader_config['max_rotation_speed'],
                                                   0, 0,
                                                   0, 0,
                                                   -self.follower_config['max_rotation_speed']), dtype=np.float32),
                                     high=np.array((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT,
                                                    self.leader_config['max_speed'], 360,
                                                    self.leader_config['max_rotation_speed'],
                                                    self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT,
                                                    self.follower_config['max_speed'], 360,
                                                    self.follower_config['max_rotation_speed']), dtype=np.float32))

    def _trajectory_in_box(self):
        """Строит массив точек маршрута Ведущего, которые входят в коробку, в которой должен находиться Ведомый."""

        self.green_zone_trajectory_points = list()

        accumulated_distance = 0

        for cur_point, prev_point in zip(reversed(self.leader_factual_trajectory[:-1]),
                                         reversed(self.leader_factual_trajectory[1:])):

            accumulated_distance += distance.euclidean(prev_point, cur_point)

            if accumulated_distance <= self.max_distance:
                self.green_zone_trajectory_points.append(cur_point)
            else:
                break

    def _get_green_zone_border_points(self):

        green_zone_points_list = self.green_zone_trajectory_points

        self.left_border_points_list = list()
        self.right_border_points_list = list()

        for cur_point, prev_point in zip(green_zone_points_list[1::2], green_zone_points_list[:-1:2]):
            move_direction = angle_to_point(prev_point, cur_point)
            point_distance = self._to_pixels(distance.euclidean(prev_point, cur_point))

            right_border_angle = angle_correction(move_direction + 90)
            left_border_angle = angle_correction(move_direction - 90)

            res_point = np.divide((cur_point + prev_point), 2)

            right_border_vec = np.array((self.max_dev * cos(radians(right_border_angle)),
                                         self.max_dev * sin(radians(right_border_angle))))
            left_border_vec = np.array((self.max_dev * cos(radians(left_border_angle)),
                                        self.max_dev * sin(radians(left_border_angle))))

            self.right_border_points_list.append(res_point + right_border_vec)
            self.left_border_points_list.append(res_point + left_border_vec)

    def _reward_computation(self):
        """функция для расчёта награды на основе конфигурации награды"""
        # Скорее всего, это можно сделать красивее
        res_reward = 0

        if self.stop_signal:
            res_reward += self.reward_config.leader_stop_penalty
        #             print("Лидер стоит по просьбе агента", self.reward_config.leader_stop_penalty)
        else:
            res_reward += self.reward_config.leader_movement_reward
        #             print("Лидер идёт по маршруту", self.reward_config.leader_movement_reward)

        if self.follower_too_close:
            res_reward += self.reward_config.too_close_penalty
        #             print("Слишком близко!", self.reward_config.too_close_penalty)
        else:
            if self.is_in_box and self.is_on_trace:
                res_reward += self.reward_config.reward_in_box
            #                 print("В коробке на маршруте.", self.reward_config.reward_in_box)
            elif self.is_in_box:
                # в пределах погрешности
                res_reward += self.reward_config.reward_in_dev
            #                 print("В коробке, не на маршруте", self.reward_config.reward_in_dev)
            elif self.is_on_trace:
                res_reward += self.reward_config.reward_on_track
            #                 print("на маршруте, не в коробке", self.reward_config.reward_on_track)
            else:
                if self.step_count > self.warm_start:
                    res_reward += self.reward_config.not_on_track_penalty
        #                 print("не на маршруте, не в коробке", self.reward_config.not_on_track_penalty)

        if self.crash:
            res_reward += self.reward_config.crash_penalty
            print("АВАРИЯ!", self.reward_config.crash_penalty)

        return res_reward

    def _check_agent_position(self):
        # если меньше, не построить траекторию
        if len(self.green_zone_trajectory_points) > 2:
            closest_point_in_box_id = self.closest_point(self.follower.position, self.green_zone_trajectory_points)
            closest_point_in_box = self.green_zone_trajectory_points[int(closest_point_in_box_id)]

            closest_green_distance = distance.euclidean(self.follower.position, closest_point_in_box)

            if closest_green_distance <= self.leader_pos_epsilon:
                self.is_on_trace = True
                self.is_in_box = True

            elif closest_green_distance <= self.max_dev:
                # Агент в пределах дистанции
                self.is_in_box = True
                self.is_on_trace = False

            else:
                closest_point_on_trajectory_id = self.closest_point(self.follower.position,
                                                                    self.leader_factual_trajectory)
                closest_point_on_trajectory = self.leader_factual_trajectory[int(closest_point_on_trajectory_id)]

                if distance.euclidean(self.follower.position, closest_point_on_trajectory) <= self.leader_pos_epsilon:
                    self.is_on_trace = True
                    self.is_in_box = False

        # Проверка вхождения в ближний круг лидера
        # TODO: учитывать лидера и следующего не как точки в идеале
        if distance.euclidean(self.leader.position, self.follower.position) <= self.min_distance:
            self.follower_too_close = True
        else:
            self.follower_too_close = False

    def _to_meters(self, pixels):
        return pixels / self.PIXELS_TO_METER

    def _to_pixels(self, meters):
        return meters * self.PIXELS_TO_METER

    def _to_seconds(self, frames):
        return frames / self.framerate

    def _to_frames(self, seconds):
        return seconds * self.framerate

    @staticmethod
    def closest_point(point, points, return_id=True):
        """Метод определяет ближайшую к точке точку из массива точек"""
        points = np.asarray(points)
        dist_2 = np.sum((points - point) ** 2, axis=1)

        if not return_id:
            return np.min(dist_2)
        else:
            return np.argmin(dist_2)


class TestGameAuto(Game):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TestGameManual(Game):
    def __init__(self):
        super().__init__(manual_control=True, add_obstacles=True, game_width=1500, game_height=1000,
                         max_steps=15000,
                         framerate=5000,
                         obstacle_number=35,
                         constant_follower_speed=False,
                         max_distance=4,
                         max_dev=1,
                         add_bear=True,
                         bear_behind=True,
                         multi_random_bears=False,
                         bear_number=1,
                         corridor_length=8,
                         corridor_width=1.5,
                         negative_speed=True,
                         follower_speed_koeff=0.6,
                         leader_speed_regime={
                             0: [0.2, 1],
                             200: 1,
                             1000: [0.5, 1],
                             1500: 0.75,
                             2000: 0,
                             2500: 1,
                             3000: [0.5, 1],
                             4000: [0.0, 0.5],
                             5000: [0.4, 1],
                         },
                         leader_acceleration_regime={0: 0,
                                                     3100: 0.03,
                                                     4500: 0},
                         multiple_end_points=False,
                         warm_start=0,
                         early_stopping={"max_distance_coef": 4, "low_reward": -300},
                         random_frames_per_step=[2, 20],
                         follower_sensors={
                             'LeaderPositionsTracker_v2': {
                                 'sensor_name': 'LeaderPositionsTracker_v2',
                                 'eat_close_points': True,
                                 'saving_period': 8,
                                 'start_corridor_behind_follower':True
                                 },
                             #'LeaderTrackDetector_vector': {
                             #    'sensor_name': 'LeaderTrackDetector_vector',
                             #    'position_sequence_length': 10},

                             'LeaderTrackDetector_radar': {
                                 'sensor_name': 'LeaderTrackDetector_radar',
                                 'position_sequence_length': 100,
                                 'radar_sectors_number': 7,
                                 'detectable_positions': 'near'},
                             "LeaderCorridor_lasers": {
                                 'sensor_name': 'LeaderCorridor_lasers',
                                 "react_to_obstacles": True,
                                 "front_lasers_count": 5,
                                 "back_lasers_count": 2,
                                 "react_to_green_zone": True,
                                 "laser_length": 150
                             },
                             "LaserSensor": {
                                 'sensor_name': 'LaserSensor',
                                 "available_angle": 360,
                                 "angle_step": 10,
                                 "points_number": 20,
                                 "sensor_range": 5,
                                 "sensor_speed": 0.1,
                                 "return_all_points":False,
                                 "return_only_distances": True
                             }
                         }
                         )


class TestGameBaseAlgoNoObst(Game):
    def __init__(self):
        super().__init__(manual_control=False, add_obstacles=False, game_width=1500, game_height=1000,
                         early_stopping={"max_distance_coef": 1.2, "low_reward": -100}
                         )


class TestGameBaseAlgoObst(Game):
    def __init__(self):
        super().__init__(manual_control=False, add_obstacles=True, game_width=1500, game_height=1000,
                         early_stopping={"max_distance_coef": 1.2, "low_reward": -100},
                         follower_sensors={"GreenBoxBorderSensor": {"sensor_range": 2,
                                                                    "available_angle": 180,
                                                                    "angle_step": 45}})


class TestGameNEAT(Game):
    def __init__(self):
        super().__init__(manual_control=False, add_obstacles=False,
                         early_stopping={"max_distance_coef": 1.2, "low_reward": -100},
                         discrete_action_space=True)


gym_register(
    id="Test-Cont-Env-Auto-v0",
    entry_point="continuous_grid_arctic.follow_the_leader_continuous_env:TestGameAuto",
    reward_threshold=10000
)

gym_register(
    id="Test-Cont-Env-Manual-v0",
    entry_point="continuous_grid_arctic.follow_the_leader_continuous_env:TestGameManual",
    reward_threshold=10000
)

gym_register(
    id="Test-Cont-Env-Auto-Follow-no-obstacles-v0",
    entry_point="continuous_grid_arctic.follow_the_leader_continuous_env:TestGameBaseAlgoNoObst")

gym_register(
    id="Test-Cont-Env-Auto-Follow-with-obstacles-v0",
    entry_point="continuous_grid_arctic.follow_the_leader_continuous_env:TestGameBaseAlgoObst")

gym_register(
    id="Test-Game-Neat-v0",
    entry_point="continuous_grid_arctic.follow_the_leader_continuous_env:TestGameNEAT")
