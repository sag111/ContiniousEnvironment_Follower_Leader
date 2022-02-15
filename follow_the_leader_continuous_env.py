import pygame
import os
from math import pi, degrees, radians, cos, sin
import numpy as np
from classes import AbstractRobot, GameObject
from scipy.spatial import distance
from reward_constructor import Reward
import gym
from gym.envs.registration import register as gym_register
from gym.spaces import Discrete,Box, Dict, Tuple

class Game(gym.Env):
    def __init__(self, game_width=1500, 
                 game_height=1000,
                 framerate=100,
                 caption="Serious Robot Follower Simulation v.-1",
                 trajectory=None,
                 leader_pos_epsilon=20,
                 show_leader_path=True,
                 show_leader_trajectory=True,
                 show_rectangles=True,
                 show_box=True,
                 simulation_time_limit=None,
                 reward_config = None,
                 pixels_to_meter = 50,
                 # 
                 min_distance = 1, # в метрах
                 max_distance = 4, # в метрах
                 max_dev = 1, # в метрах
                 warm_start = 3, # в секундах
                 manual_control = False,
                 max_steps=5000
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
            список точек, по которым едет Ведущий. Если None список генерируется случайно;
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
            путь до json-конфигурации награды, созданной с помощью класса reward_constructor. Если None, создаёт reward по умолчанию (Ivan v.1)
        pixels_to_meter (int): 
            число пикселей, который в данной модели отражает один метр;
        min_distance (int): 
            минимальная дистанция (в метрах), которую должен соблюдать Ведомый от Ведущего;
        max_distance (int): 
            максимальная дистанция (в метрах), дальше которой Ведомый не должен отставать от Ведущего (по маршруту);
        max_dev (int): 
            максимальная дистанция (в метрах), в пределах которой Ведомый может отклониться от маршрута;
        warm_start (int): 
            число секунд, в пределах которого Ведомый не будет получать штраф (сейчас не реализовано полноценно);
        manual_control (bool): 
            использовать ручное управление Ведомым;
        max_steps (int): 
            максимальное число шагов для одной симуляции.
        """
        # нужно для сохранения видео
        self.metadata = {"render.modes": ["rgb_array"]}
        # Здесь можно задать дополнительные цвета в формате RGB
        self.colours = {
                            'white':(255,255,255),
                            'black':(0,0,0),
                            'gray':(30,30,30),
                            'blue':(0,0,255),
                            'red':(255,0,0),
                            'green':(0,255,0)
                        }
        
        # TODO: сделать нормально
        metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": framerate}
        
        # задание траектории, которое полноценно обрабатывается в методе reset()
        self.trajectory = trajectory
        self.trajectory_generated = False
        
        # номер симуляции
        self.simulation_number = 0
        
        self.DISPLAY_WIDTH=game_width
        self.DISPLAY_HEIGHT=game_height
        self.PIXELS_TO_METER=pixels_to_meter
        self.framerate = framerate
        
        self.leader_pos_epsilon = leader_pos_epsilon
        
        # Настройки визуализации
        self.show_leader_path = show_leader_path
        self.show_leader_trajectory = show_leader_trajectory
        self.show_rectangles = show_rectangles
        self.show_box = show_box
        
        self.simulation_time_limit = simulation_time_limit
        
        if reward_config:
            self.reward_config = Reward.from_json(reward_config)
        else:
            self.reward_config = Reward(leader_movement_reward=0)
        
        self.overall_reward = 0
        
        self.min_distance = min_distance * self.PIXELS_TO_METER
        self.max_distance = max_distance * self.PIXELS_TO_METER
        self.max_dev = max_dev * self.PIXELS_TO_METER
        
        self.warm_start = warm_start*framerate
        
        self.leader_img =  pygame.image.load("{}/imgs/car_yellow.png".format(os.path.dirname(os.path.abspath(__file__))))
        self.follower_img = pygame.image.load("{}/imgs/car_poice.png".format(os.path.dirname(os.path.abspath(__file__))))
        
        self.caption = caption
        
        self.manual_control = manual_control
        
        self.max_steps = max_steps

        
        self.reset()
        
        # Пространство действий - мы определяем два действительных числа - скорость агента и угловую скорость агента;
        # self.leader - ведущий, self.follower - ведомый, создаются в функции reset.
        
        self.action_space = Box(np.array((self.follower.min_speed,-self.follower.max_rotation_speed), dtype=np.float32),
                                np.array((self.follower.max_speed,self.follower.max_rotation_speed), dtype=np.float32))
        
        # именно в таком виде, потому что stable_baselines не может векторизовать space, представленный в виде словаря.
        # скорее всего, можно в перспективе использовать какой-то флаттенер, потому что пока что это курам на смех.
        # ниже закомменчен вариант с Dict в качестве observation_space
        
        self.observation_space = Box(np.array([0,0,
                                              self.leader.min_speed,
                                              0,
                                              -self.leader.max_rotation_speed,
                                              0,0,
                                              self.follower.min_speed,
                                              0,
                                              -self.follower.max_rotation_speed,
                                              self.min_distance,
                                              self.max_distance,
                                              self.max_dev], dtype=np.float32),
                                     np.array([self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT,
                                              self.leader.max_speed,
                                              360,
                                              self.leader.max_rotation_speed,
                                              self.DISPLAY_WIDTH,
                                              self.DISPLAY_HEIGHT,
                                              self.follower.max_speed,
                                              360,
                                              self.follower.max_rotation_speed,
                                              self.min_distance,
                                              self.max_distance,
                                              self.max_dev], dtype=np.float32
                                              ))
        
#         self.observation_space = Dict({
#                "leader_location_x": Box(0,self.DISPLAY_WIDTH, shape=[1]),
#                "leader_location_y": Box(0,self.DISPLAY_HEIGHT, shape=[1]),
#                "leader_speed": Box(self.leader.min_speed,self.leader.max_speed, shape=[1]),
#                "leader_direction": Discrete(360),
#                "leader_rotation_speed": Box(-self.leader.max_rotation_speed,self.leader.max_rotation_speed,shape=[1]), 
#                "follower_location_x": Box(0,self.DISPLAY_WIDTH, shape=[1]),
#                "follower_location_y": Box(0,self.DISPLAY_HEIGHT, shape=[1]),
#                "follower_speed": Box(self.follower.min_speed,self.follower.max_speed,shape=[1]), 
#                "follower_direction": Discrete(360), 
#                "follower_rotation_speed": Box(-self.follower.max_rotation_speed,self.follower.max_rotation_speed, shape=[1]), 
#                                       })

        
    def reset(self):
        """Стандартный для gym обработчик инициализации новой симуляции. Возвращает инициирующее наблюдение."""
        
        print("===Запуск симуляции номер {}===".format(self.simulation_number))
        # Создание ведущего и ведомого
        # TODO: сторонние конфигурации для создания роботов
        self.leader = AbstractRobot("leader",
                            image=self.leader_img,
                            height = 0.38*self.PIXELS_TO_METER,
                            width = 0.52*self.PIXELS_TO_METER,
                            min_speed=0,
                            max_speed=0.5*self.PIXELS_TO_METER/100,
                            max_speed_change=0.005*self.PIXELS_TO_METER/100,
                            max_rotation_speed=57.296/100,
                            max_rotation_speed_change=20/100,
                            start_position=(self.DISPLAY_WIDTH/2, self.DISPLAY_HEIGHT/2))
        
        self.follower = AbstractRobot("follower",
                              image=self.follower_img,
                              height = 0.5 * self.PIXELS_TO_METER,
                              width = 0.35 * self.PIXELS_TO_METER,
                              min_speed=0,
                              max_speed=0.5*self.PIXELS_TO_METER/100,
                              max_speed_change=0.005*self.PIXELS_TO_METER/100,
                              max_rotation_speed=57.296/100,
                              max_rotation_speed_change=20/100,   
                              start_position=((self.DISPLAY_WIDTH/2)+50, (self.DISPLAY_HEIGHT/2)+50))

        # список точек пройденного пути Ведущего, которые попадают в границы требуеимого расстояния
        self.green_zone_trajectory_points = list()
        
        # в случае, если траектория не задана или была сгенерированна, при каждой симуляции генерируем новую случайную траекторию
        if (self.trajectory is None) or self.trajectory_generated:
            self.trajectory=self.generate_trajectory()
            self.trajectory_generated = True
        
        self.trajectory=self.trajectory
        
        # Добавление начальной позиции лидера к траектории, чтобы отображать линию и от его начала к первой точке
        self.trajectory.insert(0, self.leader.start_position)
        
        # Флаги для расчёта reward
        self.stop_signal = False
        self.is_in_box = False
        self.is_on_trace = False
        self.step_count = 0
        self.follower_too_close = False
        self.crash = False
        
        # Список всех игровых объектов
        # Предполагается, что препятствия будут добавляться сюда
        self.game_object_list = list()
        self.game_object_list.append(self.leader)
        self.game_object_list.append(self.follower)
        
        # Флаг конца симуляции
        self.done = False
         
        self.cur_target_id = 1 # индекс целевой точки из маршрута
        self.leader_factual_trajectory = list() # список, который сохраняет пройденные лидером точки;
        self.leader_finished = False # флаг, показывает, закончил ли лидер маршрут, т.е. достиг ли последней точки
        self.cur_target_point = self.trajectory[self.cur_target_id] # координаты текущей целевой точки (возможно избыточны)
        
        # Инициализация сеанса pygame, создание окна и часов
        # Возможно не нужно пересоздавать окно каждый раз, стоит подумать
        pygame.init()
#         pygame.font.init()
        self.gameDisplay = pygame.display.set_mode((self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT))
        pygame.display.set_caption(self.caption)
        self.clock = pygame.time.Clock()
        
        # обнуление параметров, которые отслеживаются в рамках симуляции
        self.step_count = 0 # число шагов
        self.overall_reward = 0 # суммарная награда за все шаги
        
        self.simulation_number += 1
        
        return self._get_obs()
    

    def step(self, action):
        """Стандартный для gym обработчик одного шага среды (в данном случае один кадр)"""
        self.is_in_box = False
        self.is_on_trace = False
        
        # Если контролирует автомат, то нужно преобразовать угловую скорость с учётом её знака.
        if self.manual_control:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                if self.manual_control:
                    self.manual_game_contol(event,self.follower)
        else:
            self.follower.command_forward(action[0])
            #self.follower.rotation_speed = action[1]  # command_turn ведь плавно изменяет текущую скорость поворота в зависимости от ограничения на изменение, зачем её явно задавать?
            if action[1]<0:
                self.follower.command_turn(abs(action[1]),-1)
            elif action[1]>0:
                self.follower.command_turn(action[1],1)
            else:
                self.follower.command_turn(0,0)
            
        self.follower.move()
        # TODO:проверка на столкновение с препятствием вероятно здесь[Слава]
            
        
        # Определение коробки и агента в ней
        # определение текущих точек маршрута, которые являются подходящими для Агента
        self.green_zone_trajectory_points = list()
        self._trajectory_in_box()
        
        # определяем положение Агента относительно маршрута и коробки
        
        self._check_agent_position()
        
        # работа с движением лидера
        prev_leader_position = self.leader.position.copy()

        if distance.euclidean(self.leader.position, self.cur_target_point) < self.leader_pos_epsilon:
            self.cur_target_id+=1
            if self.cur_target_id >= len(self.trajectory):
                self.leader_finished = True
            else:
                self.cur_target_point = self.trajectory[self.cur_target_id]

        if not self.leader_finished:
            self.leader.move_to_the_point(self.cur_target_point)
        else:
            self.leader.command_forward(0)
            self.leader.command_turn(0,0)
            
        # TODO: обработка столкновений лидера [Слава]
           
        # чтобы не грузить записью КАЖДОЙ точки, записываем точку раз в 5 миллисекунд;
        # TODO: сделать параметром;
        
        if pygame.time.get_ticks()%5==0:
            self.leader_factual_trajectory.append(self.leader.position.copy())

        # обработка аварий агента в случае столкновения с лидером или границами карты
        if self.leader.rectangle.colliderect(self.follower.rectangle) or \
            any(self.follower.position>=(self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)) or any(self.follower.position<=(0, 0)):
            self.crash=True
            self.done=True
            
        
        res_reward = self._reward_computation()
        
#         if (pygame.time.get_ticks()<self.warm_start) and (res_reward < 0):
#             res_reward = 0
        self.overall_reward += res_reward
        
        self.clock.tick(self.framerate)
        
        if self.simulation_time_limit is not None:
            if pygame.time.get_ticks()*1000 > self.simulation_time_limit:
                self.done=True
                print("Время истекло! Прошло {} секунд.".format(self.simulation_time_limit))
        
        obs = self._get_obs()
        
        self.step_count+=1
        
        if self.step_count > self.max_steps:
            self.done=True
#         print("Аккумулированная награда на step {0}: {1}".format(self.step_count, self.overall_reward))
#         print()
        
        return obs, res_reward, self.done, {}
    
    
    def render(self, custom_message=None, **kwargs):
        """Стандартный для gym метод отображения окна и обработки событий в нём (например, нажатий клавиш)"""
        
        self._show_tick()
        pygame.display.update()
        
        return np.transpose(
                pygame.surfarray.array3d(self.gameDisplay), axes=(1, 0, 2))
        
    def rotate_object(self,object_to_rotate):
        """Поворачивает изображение объекта, при этом сохраняя его центр и прямоугольник для взаимодействия.
        """
        cur_rect = object_to_rotate.rectangle
        # Rotate the original image without modifying it.
        new_image = pygame.transform.rotate(object_to_rotate.image, -object_to_rotate.direction)
        # Get a new rect with the center of the old rect.
        object_to_rotate.rectangle = new_image.get_rect(center=cur_rect.center)
        
        return new_image
    
    
    def show_object(self,object_to_show):
        """Отображает объект с учётом его направления"""
        cur_image = object_to_show.image
        if hasattr(object_to_show, "direction"):
            cur_image = self.rotate_object(object_to_show)
        
        self.gameDisplay.blit(cur_image, (object_to_show.position[0]-object_to_show.width/2, object_to_show.position[1]-object_to_show.height/2))
        object_to_show.rectangle = cur_image.get_rect(center=object_to_show.position)
        
        if self.show_rectangles:
            pygame.draw.rect(self.gameDisplay,self.colours["red"],object_to_show.rectangle,width=1)
            
            
            
    def _show_tick(self):
        """Отображает всё, что положено отображать на каждом шаге"""
        self.gameDisplay.fill(self.colours["white"]) # фон
        
        # отображение полного маршрута Ведущего
        if self.show_leader_path:
            pygame.draw.aalines(self.gameDisplay,self.colours["red"],False,self.trajectory)
        
        # отображение зоны, в которой нужно находиться Ведомому
        if self.show_box:
            if len(self.green_zone_trajectory_points)>5:
                green_line = pygame.draw.polygon(self.gameDisplay,self.colours["green"],self.green_zone_trajectory_points[::5], width=self.max_dev*2)
        
        # отображение пройденной Ведущим траектории
        if self.show_leader_trajectory:
            for cur_point in self.leader_factual_trajectory[::10]: # Каждую 10ю точку показываем.
                pygame.draw.circle(self.gameDisplay, self.colours["black"], cur_point, 3)
        
        # отображение всех игровых объектов, которые были добавлены в список игровых объектов
        for cur_object in self.game_object_list:
            self.show_object(cur_object)
        
        # TODO: здесь будет отображение препятствий (лучше, если в рамках цикла выше, то есть как игровых объектов) [Слава]
        
        
        # отображение круга минимального расстояния
        if self.follower_too_close:
            close_circle_width = 2
        else:
            close_circle_width = 1
            
        self.leader_close_circle = pygame.draw.circle(self.gameDisplay, self.colours["red"], self.leader.position, self.min_distance, width=close_circle_width)
        
        
        
        
    def generate_trajectory(self, n=3, min_distance = 50, border = 20, iter_limit = 10000):
        """Случайно генерирует точки на карте, по которым должен пройти ведущий"""
        #TODO: добавить проверку, при которойо точки не на одной прямой
        #TODO: добавить отдельную функцию, которая использует эту: 
        # на вход принимает шаблон -- список из r и c, где 
        #    r -- placeholder, на место которого будут подставляться случайные точки
        #    c -- координаты точки, которые точно должны присутствовать в пути (например, координаты "моста")
        #TODO: вообще нужен отдельный класс для траекторий;
        #TODO: если строить маршрут с учётом препятствий сразу, вероятно обработка будет здесь или где-то рядом [Слава]
        #TODO: ограничение на число итераций цикла (иначе может уйти в бесконечность).
        #вероятно нужно сделать staticmethod
        
        trajectory = list()
        
        i = 0 # пока отслеживаем зацикливание по числу итераций на генерацию каждой точки. Примитивно, но лучше, чем никак
        
        while (len(trajectory) < n) and (i < iter_limit):
            new_point = np.array((np.random.randint(border,high=self.DISPLAY_WIDTH-border),
                                  np.random.randint(border,high=self.DISPLAY_HEIGHT-border)))
            
            if len(trajectory)==0:
                trajectory.append(new_point)
                i=0
            else:
                to_add = True
                
                # работает только на ограниченном числе точек, может уйти в бесконечный цикл, осторожнее!!!
                
                for prev_point in trajectory:
                    if distance.euclidean(prev_point,new_point) < min_distance:
                        to_add=False
                
                if to_add:
                    trajectory.append(new_point)  
                
                i+=1
            
        return trajectory
    
    
    def manual_game_contol(self, event, follower):
        """обработчик нажатий клавиш при ручном контроле."""
        # В теории, можно на основе этого класса сделать управляемого руками Ведущего. Но надо модифицировать.
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                if follower.rotation_direction > 0:
                    follower.rotation_speed=0
                    follower.rotation_direction=0
                    follower.command_turn(0,0)
                else:
                    follower.command_turn(follower.rotation_speed+2,-1)


            if (event.key == pygame.K_RIGHT):
                if follower.rotation_direction < 0:
                    follower.rotation_speed=0
                    follower.rotation_direction=0
                    follower.command_turn(0,0)
                else:
                    follower.command_turn(follower.rotation_speed+2,1)


            if event.key == pygame.K_UP:
                follower.command_forward(follower.speed+self.PIXELS_TO_METER)

            if event.key == pygame.K_DOWN:
                follower.command_forward(follower.speed-self.PIXELS_TO_METER)
    
    
    def _get_obs(self):
        """Возвращает наблюдения (observations) среды каждый шаг (step)"""
        return np.array([self.leader.position[0],
                         self.leader.position[1],
                         self.leader.speed,
                         self.leader.direction,
                         self.leader.rotation_speed,
                         self.follower.position[0],
                         self.follower.position[1],
                         self.follower.speed,
                         self.follower.direction,
                         self.follower.rotation_speed,
                         self.min_distance,
                         self.max_distance,
                         self.max_dev],dtype=np.float32)
    
#                 {#"trajectory": self.leader_factual_trajectory, 
#                "leader_location_x": self.leader.position[0],
#                "leader_location_y": self.leader.position[1],
#                "leader_speed":self.leader.speed,
#                "leader_direction":int(self.leader.direction),
#                "leader_rotation_speed":self.leader.rotation_speed,
#                "follower_location_x":self.follower.position[0],
#                "follower_location_y":self.follower.position[1],
#                "follower_speed":self.follower.speed,
#                "follower_direction":int(self.follower.direction),
#                "follower_rotation_speed":self.follower.rotation_speed,}.values()
        
    
    def _trajectory_in_box(self):
        """Строит массив точек маршрута Ведущего, которые входят в коробку, в которой должен находиться Ведомый."""
        
        self.green_zone_trajectory_points = list()
        
        accumulated_distance = 0
        
        for cur_point, prev_point in zip(reversed(self.leader_factual_trajectory[:-1]),reversed(self.leader_factual_trajectory[1:])):
            
            accumulated_distance+=distance.euclidean(prev_point,cur_point)
            
            if accumulated_distance<=self.max_distance: # /self.PIXELS_TO_METER
                self.green_zone_trajectory_points.append(cur_point)
            else:
                break
        
     
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
            closest_point_in_box_id = self.closest_point(self.follower.position,self.green_zone_trajectory_points)
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
                closest_point_on_trajectory_id = self.closest_point(self.follower.position,self.leader_factual_trajectory)
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
        pass
    
    def _to_pixels(self, meters):
        pass
    
    def _to_seconds(self, frames):
        pass
    
    def _to_frames(self, seconds):
        pass

    @staticmethod
    def closest_point(point, points, return_id=True):
        """Метод определяет ближайшую к точке точку из массива точек"""
        points = np.asarray(points)
        dist_2 = np.sum((points - point)**2, axis=1)
        
        if not return_id:
            return np.min(dist_2)
        else:
            return np.argmin(dist_2)            

        
    
class TestGameAuto(Game):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
class TestGameManual(Game):
    def __init__(self):
        super().__init__(manual_control=True)

        
gym_register(
    id="Test-Cont-Env-Auto-v0",
    entry_point="follow_the_leader_continuous_env:TestGameAuto",
    reward_threshold=10000
)

gym_register(
    id="Test-Cont-Env-Manual-v0",
    entry_point="follow_the_leader_continuous_env:TestGameManual",
    reward_threshold=10000
)

