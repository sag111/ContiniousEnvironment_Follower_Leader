import pygame
from math import pi, degrees, radians, cos, sin
import numpy as np
from classes import AbstractRobot, GameObject
from scipy.spatial import distance
from reward_constructor import Reward
import gym
from gym.envs.registration import register as gym_register
from gym.spaces import Box

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
                 manual_control = False
                ):
        
        self.colours = {
                            'white':(255,255,255),
                            'black':(0,0,0),
                            'gray':(30,30,30),
                            'blue':(0,0,255),
                            'red':(255,0,0),
                            'green':(0,255,0)
                        }
        
        self.trajectory = trajectory
        self.trajectory_generated = False

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
            self.reward_config = Reward()
        
        self.overall_reward = 0
        
        self.min_distance = min_distance * self.PIXELS_TO_METER
        self.max_distance = max_distance * self.PIXELS_TO_METER
        self.max_dev = max_dev * self.PIXELS_TO_METER
        
        self.warm_start = warm_start*1000
        
        self.leader_img =  pygame.image.load("imgs/car_yellow.png")
        self.follower_img = pygame.image.load("imgs/car_poice.png")
        
        self.caption = caption
        
        self.manual_control = manual_control
        
        self.reset()
        
        self.action_space = Box(np.array([self.follower.min_speed,self.follower.max_speed]),np.array([-self.follower.max_rotation_speed,self.follower.max_rotation_speed]))
    
    
    def reset(self):
        print("===Запуск симуляции номер {}===".format(self.simulation_number))
        
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


        self.green_zone_trajectory_points = list()
        
        if (self.trajectory is None) or self.trajectory_generated:
            self.trajectory=self.generate_trajectory()
            self.trajectory_generated = True
        
        self.trajectory=self.trajectory
        
        # Флаги для расчёта reward
        self.stop_signal = False
        self.is_in_box = False
        self.is_on_trace = False
        self.step_count = 0
        self.follower_too_close = False
        self.crash = False
        
        self.game_object_list = list()
        self.game_object_list.append(self.leader)
        self.game_object_list.append(self.follower)
        
        self.done = False
        
        self.trajectory.insert(0, self.leader.start_position)
        
        self.cur_target_id = 0
        self.leader_factual_trajectory = list()
        self.leader_finished = False
        self.cur_target_point = self.trajectory[1]
        
        pygame.init()
        self.gameDisplay = pygame.display.set_mode((self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT))
        pygame.display.set_caption(self.caption)
        self.clock = pygame.time.Clock()
    
    
    def rotate_object(self,object_to_rotate):
        """Rotate the image while keeping its center."""
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
        """Отображает всё, что положено отображать на каждом тике"""
        self.gameDisplay.fill(self.colours["white"])
        
        if self.show_leader_path:
            pygame.draw.aalines(self.gameDisplay,self.colours["red"],False,self.trajectory)
        
        if self.show_box:
            if len(self.green_zone_trajectory_points)>5:
                green_line = pygame.draw.polygon(self.gameDisplay,self.colours["green"],self.green_zone_trajectory_points[::5], width=self.max_dev*2)
        
        
        if self.show_leader_trajectory:
            for cur_point in self.leader_factual_trajectory[::10]: # Каждую 10ю точку показываем.
                pygame.draw.circle(self.gameDisplay, self.colours["black"], cur_point, 3)

        for cur_object in self.game_object_list:
            self.show_object(cur_object)
        
        # Круг, ближе которого не стоит приближаться
        
        leader_close_circle = pygame.draw.circle(self.gameDisplay, self.colours["red"], self.leader.position, self.min_distance, width=1)
        collide = leader_close_circle.colliderect(self.follower.rectangle)
        self.follower_too_close=False
        if collide:
            leader_close_circle = pygame.draw.circle(self.gameDisplay, self.colours["red"], self.leader.position, self.min_distance, width=2)
            self.follower_too_close=True
        # Это что же, логика в функции рисования? Отвратительно...
        
        
        
    def generate_trajectory(self, n=3, min_distance = 50, border = 20):
        """Генерирует точки на карте, по которым должен пройти ведущий"""
        #TODO: добавить проверку, при которойо точки не на одной прямой
        #Staticmethod?
        trajectory = list()
        
        while len(trajectory) < n:
            new_point = np.array((np.random.randint(border,high=self.DISPLAY_WIDTH-border),
                                  np.random.randint(border,high=self.DISPLAY_HEIGHT-border)))
            
            if len(trajectory)==0:
                trajectory.append(new_point)
            else:
                to_add = True
                
                for prev_point in trajectory:
                    if distance.euclidean(prev_point,new_point) < min_distance:
                        to_add=False
                
                if to_add:
                    trajectory.append(new_point)            
            
        return trajectory
    
    
    def manual_game_contol(self, event, follower):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
    #                 follower.rotation_direction=-1
                if follower.rotation_direction > 0:
                    follower.rotation_speed=0
                    follower.rotation_direction=0
                else:
                    follower.rotation_direction=-1
                    follower.rotation_speed+=2
                follower.command_turn(follower.rotation_speed,-1)
                print("agent rotation speed and rotation direction", follower.rotation_speed, follower.rotation_direction)
                print("current follower direction: ", follower.direction)


            if (event.key == pygame.K_RIGHT):
    #                 follower.rotation_direction = 1
                if follower.rotation_direction < 0:
                    follower.rotation_speed=0
                    follower.rotation_direction=0
                else:
                    follower.rotation_direction=1
                    follower.rotation_speed+=2
                follower.command_turn(follower.rotation_speed,1)
                print("agent rotation speed and rotation direction", follower.rotation_speed, follower.rotation_direction)
                print("current follower direction: ", follower.direction)


            if event.key == pygame.K_UP:
                follower.command_forward(follower.speed+self.PIXELS_TO_METER)
                print("agent speed", follower.speed)

            if event.key == pygame.K_DOWN:
                follower.command_forward(follower.speed-self.PIXELS_TO_METER)
                print("agent speed", follower.speed)
            
        
    def render(self, custom_message=None, **kwargs):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            if self.manual_control:
                self.manual_game_contol(event,self.follower)

        self._show_tick()
        pygame.display.update()

        
    
    def step(self, action):
        
        self.is_in_box = False
        self.is_on_trace = False
        
        if not self.manual_control:
            self.follower.command_forward(action[0])
            if action[1]<0:
                self.follower.command_turn(abs(action[1]),-1)
            elif action[1]>0:
                self.follower.command_turn(action[1],1)
            else:
                self.follower.command_turn(0,0)
                
            
        self.follower.move()
            
        # Определение коробки и агента в ней
        # Вынести в отдельную функцию
        
        self.green_zone_trajectory_points = list()
        self._trajectory_in_box()
        
        if len(self.green_zone_trajectory_points) > 2:
            closest_point_in_box_id = self.closest_point(self.follower.position,self.green_zone_trajectory_points)
            closest_point_in_box = self.green_zone_trajectory_points[int(closest_point_in_box_id)]

            closest_green_distance = distance.euclidean(self.follower.position, closest_point_in_box)
#             TODO: перенести в функцию рисования
#             pygame.draw.line(self.gameDisplay, self.colours["blue"], self.follower.position, closest_point_in_box)

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
#                 TODO: перенести в функцию рисования
#                 pygame.draw.line(self.gameDisplay, self.colours["black"], self.follower.position, closest_point_on_trajectory)
                
                if distance.euclidean(self.follower.position, closest_point_on_trajectory) <= self.leader_pos_epsilon:
                    self.is_on_trace = True
                    self.is_in_box = False

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

        if pygame.time.get_ticks()%5==0:
            self.leader_factual_trajectory.append(self.leader.position.copy())

        if self.leader.rectangle.colliderect(self.follower.rectangle) or \
            any(self.follower.position>=(self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT)) or any(self.follower.position<=(0, 0)):
            self.crash=True
            self.done=True
            
        res_reward = self._reward_computation()
        
        if (pygame.time.get_ticks()>self.warm_start) and (res_reward < 0):
            res_reward = 0
        self.overall_reward += res_reward
        
        self.clock.tick(self.framerate)

        if self.simulation_time_limit is not None:
            if pygame.time.get_ticks()*1000 > self.simulation_time_limit:
                self.done=True
                print("Время истекло! Прошло {} секунд.".format(self.simulation_time_limit))
        
        obs = {"trajectory": self.leader_factual_trajectory, 
               "leader_location": self.leader.position,
               "leader_speed":self.leader.speed,
               "leader_direction":self.leader.direction,
               "leader_rotation_speed":self.leader.rotation_speed,
               "follower_location":self.follower.position,
               "follower_speed":self.follower.speed,
               "follower_direction":self.follower.direction,
               "follower_rotation_speed":self.follower.rotation_speed,}
        
        self.step_count+=1
        
        
#         pygame.time.wait(1000)
        
        print("Аккумулированная награда на step {0}: {1}".format(self.step_count, self.overall_reward))
        print()
        
#         pygame.time.wait(5000)
            
        return obs, res_reward, self.done, {}
    
    
    
    def _trajectory_in_box(self):
        
        self.green_zone_trajectory_points = list()
        
        accumulated_distance = 0
        
        for cur_point, prev_point in zip(reversed(self.leader_factual_trajectory[:-1]),reversed(self.leader_factual_trajectory[1:])):
            
            accumulated_distance+=distance.euclidean(prev_point,cur_point)
            
            if accumulated_distance<=self.max_distance: # /self.PIXELS_TO_METER
                self.green_zone_trajectory_points.append(cur_point)
            else:
                break
        
        
        
    def _reward_computation(self):
        # Скорее всего, это можно сделать красивее
        res_reward = 0
        
        
        if self.stop_signal:
            res_reward += self.reward_config.leader_stop_penalty
            print("Лидер стоит по просьбе агента", self.reward_config.leader_stop_penalty)
        else:
            res_reward += self.reward_config.leader_movement_reward
            print("Лидер идёт по маршруту", self.reward_config.leader_movement_reward)
        
        if self.follower_too_close:
            res_reward += self.reward_config.too_close_penalty 
            print("Слишком близко!", self.reward_config.too_close_penalty)
        else:
            if self.is_in_box and self.is_on_trace:
                res_reward += self.reward_config.reward_in_box
                print("В коробке на маршруте.", self.reward_config.reward_in_box)
            elif self.is_in_box:
                # в пределах погрешности
                res_reward += self.reward_config.reward_in_dev
                print("В коробке, не на маршруте", self.reward_config.reward_in_dev)
            elif self.is_on_trace:
                res_reward += self.reward_config.reward_on_track
                print("на маршруте, не в коробке", self.reward_config.reward_on_track)
            else:
                if self.step_count > self.warm_start:
                    res_reward += self.reward_config.not_on_track_penalty
                print("не на маршруте, не в коробке", self.reward_config.not_on_track_penalty)
        
        
#             print("Слишком близко!", self.reward_config.too_close_penalty)
        
#         leader_agent_diff_vec = abs(self.agent_pos-self.leader.cur_pos)
    
#         # Определяем близость так, а не по расстоянию Миньковского, чтобы ученсть близость по диагонали
#         if (leader_agent_diff_vec[0]<=self.min_distance) and (leader_agent_diff_vec[1] <= self.min_distance):
# #         if sum(abs(self.agent_pos - self.leader.cur_pos)) <= self.min_distance:
#             res_reward += self.reward_config.too_close_penalty 
#             print("Слишком близко!", self.reward_config.too_close_penalty)
        
        if self.crash:
            res_reward += self.reward_config.crash_penalty
            print("АВАРИЯ!", self.reward_config.crash_penalty)
        
        return res_reward
    
    @staticmethod
    def closest_point(point, points, return_id=True):
        points = np.asarray(points)
        dist_2 = np.sum((points - point)**2, axis=1)
        
        if not return_id:
            return np.min(dist_2)
        else:
            return np.argmin(dist_2)

    
class TestGame(Game):
    def __init__(self):
        super().__init__()
        
gym_register(
    id="Test-Cont-Env-v0",
    entry_point="follow_the_leader_continuous_env:TestGame",
    reward_threshold=10000
)