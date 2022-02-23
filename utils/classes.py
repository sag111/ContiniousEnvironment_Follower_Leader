import pygame
from math import pi, degrees, radians, cos, sin, atan, acos, asin, sqrt
import numpy as np
from scipy.spatial import distance
import json

class GameObject():
    def __init__(self,
                 name,
                 image=None,
                 start_position=None,
                 height=None,
                 width=None,
                 blocks_vision=True):
        """Класс, который отражает любой игровой объект и должен наследоваться конкретизирующими классами.
        Входные параметры:
        image (pygame.image):
            изображение объекта;
        start_position (tuple(int,int)):
            стартовые координаты объекта;
        height, width (int):
            размеры объекта в пикселях;
        blocks_vision (bool):
            блокирует ли объект линию видимости (для лидаров и ухода за поворот, пока не используется)
        """
        self.name = name
        self.image = image
        
        if height and width:
            self.height = height
            self.width = width
            self.image = pygame.transform.scale(self.image, (width,height))
        else:
            self.height = self.image.get_height()
            self.width = self.image.get_width()
            
            
        self.start_position = np.array(start_position,dtype=np.float32)
        self.position = self.start_position
        
        self.rectangle = self.image.get_rect(center=self.position, width=width, height=height)
        
        self.blocks_vision = blocks_vision
        
    def put(self,position):
        self.position = np.array(position,dtype=np.float32)
    
        
        

class AbstractRobot(GameObject):
    def __init__(self, 
                 name,
                 image=None,
                 start_position=None,
                 height=None, # в метрах
                 width=None, # в метрах
                 min_speed=0., # в метрах в секунду
                 max_speed=2., # в метрах в секунду
                 max_rotation_speed=60, # в градусах
                 max_speed_change=0.5, # в метрах в секунду
                 max_rotation_speed_change=57, # в градусах
                 start_direction = 0, # в градусах
                 sensor = None,
                 blocks_vision=True,
                 **kwargs
                 ):
        """Класс, который реализует робота."""
        
        super(AbstractRobot, self).__init__(name,
                 image=image,
                 start_position=start_position,
                 height=height,
                 width=width)
        
        self.blocks_vision = blocks_vision
        self.speed = 0.
        self.rotation_speed = 0.
        self.rotation_direction = 0
        # 0 -- прямо, -1 -- влево, +1 -- вправо.
        
        self.direction = start_direction  # в градусах!
        
        self.desirable_rotation_speed = 0.  # в радианах
        self.desirable_rotation_direction = 0.
        self.desirable_speed = 0.
        
        self.min_speed = min_speed
        self.max_speed = max_speed
        
        self.max_rotation_speed = max_rotation_speed
        
        self.max_speed_change = max_speed_change
        self.max_rotation_speed_change = max_rotation_speed_change
        
        self.width = width
        self.height = height
        
        if sensor is None:
            self.has_sensor = False
            self.sensor_list = []
        else:
            self.has_sensor = True
            
            if sensor is list:
                self.sensor_list = sensor
            else:
                self.sensor_list = [sensor]
            
            self._sensor_pos_update()
    
    def _sensor_pos_update(self):
        for cur_sensor in self.sensor_list:
            cur_sensor.sensor_direction = self.direction
            cur_sensor.sensor_position = self.position
    
#     def set_sensor(self, sensor_type, **kwargs):
#         new_sensor = sensor_type(self,**kwargs)
#         self.sensor_list.append(new_sensor)
    
    def command_turn(self, desirable_rotation_speed, rotation_direction):
        """Обработка команд, влияющих на скорость угловую w"""
            # Не превышаем максимальной скорости
        
        self.desirable_rotation_speed = min(desirable_rotation_speed,self.max_rotation_speed)
        self.desirable_rotation_direction = rotation_direction
        
        if (rotation_direction == 0) and (desirable_rotation_speed!=0):
            raise ValueError("Указана скорость поворота, но направление = 0!")
        
        
    def command_forward(self, desirable_speed):
        """Обработка команд, влияющих на скорость v"""
        if desirable_speed > self.max_speed:
            desirable_speed = self.max_speed
        
        if desirable_speed < self.min_speed:
            desirable_speed = self.min_speed
        
        self.desirable_speed = desirable_speed
    
    
    def _controller_call(self):
        """Изменение скорости в зависимости от установленных желаемых скоростей на основе управеения"""
        self._turn_processing()
        self._speed_processing()
     
    
    def _turn_processing(self):
        """Обработка изменения скорости поворота в такт для контроллера"""
        if self.rotation_direction==0:
            self.rotation_direction = self.desirable_rotation_direction

        if self.rotation_direction == self.desirable_rotation_direction:
            needed_change = abs(self.rotation_speed - self.desirable_rotation_speed)
            speed_rotation_change = min((needed_change, self.max_rotation_speed_change))
            
            if self.desirable_rotation_speed < self.rotation_speed:
                speed_rotation_change = -1*speed_rotation_change
        else:
            needed_change = abs(self.desirable_rotation_speed + self.rotation_speed)
            speed_rotation_change = -min((needed_change, self.max_rotation_speed_change))

        new_rotation_speed = self.rotation_speed + speed_rotation_change

        if new_rotation_speed < 0:
            self.rotation_direction = -1*self.rotation_direction
        self.rotation_speed = abs(new_rotation_speed)
        
            
    def _speed_processing(self):
        """Обработка изменения скорости в такт для контроллера"""
        needed_change = abs(self.speed-self.desirable_speed)
        speed_change  = min(self.max_speed_change, needed_change)
        
        if self.speed > self.desirable_speed:
            speed_change=-1*speed_change
        
        self.speed = self.speed + speed_change
        
        
    def move(self):
        """Функция, которая перемещает робота с учётом установленных желаемых скоростей."""
        # скорректировали скорости
        self._controller_call()
        if self.rotation_speed!=0:
            self.direction = angle_correction(self.direction + self.rotation_direction*self.rotation_speed)
        
        movement_vec = np.array((cos(radians(self.direction))*self.speed, sin(radians(self.direction))*self.speed),dtype=np.float32)
        self.position+=movement_vec
        self._sensor_pos_update()
    
    
    def move_to_the_point(self, next_point):
        """Функция автоматического управления движением к точке"""
        #TODO: сделать более хороший алгоритм следования маршруту [Слава]
        
        desirable_angle = angle_to_point(self.position,next_point)
        
        delta_turn = int(self.direction-desirable_angle)
        
        if delta_turn > 0.:
            new_rotation_direction = -1
        elif delta_turn < 0.:
            new_rotation_direction = 1
        else:
            new_rotation_direction = 0
        
        delta_turn = abs(delta_turn)

        self.command_forward(distance.euclidean(self.position,next_point))
        self.command_turn(abs(delta_turn), new_rotation_direction)
        
        self.move()
        
    def use_sensor(self, env):
        if self.has_sensor:
            
            res_scans = dict()
            
            for cur_sensor in self.sensor_list:
                cur_sensor.direction = self.direction
                cur_sensor.position = self.position
                
                if cur_sensor.sensor_name in res_scans:
                    #
                    raise KeyError("Дублирование имени сенсора: {}".format("cur_sensor.sensor_name"))
                else:
                    res_scans[cur_sensor.sensor_name] = cur_sensor.scan(env)
                
#                 res_scans.append(cur_sensor.scan(env))
            
            return res_scans
            
        else:
            return list()
            print("Нет сенсора, чтобы использовать!")
        
        
        
class LaserSensor():
    """Реализует один лазерный сенсор лидара"""
    def __init__(self,
                 sensor_direction=None,
                 sensor_position=None,
                 sensor_name="lidar",
                 available_angle=360, 
                 angle_step=10, # в градусах
                 points_number=20, # чиcло точек
                 sensor_range=5, # в метрах
                 return_all_points = False,
                 **kwargs
                ):
         
        self.sensor_name = sensor_name
        self.position = sensor_position
        self.direction = sensor_direction

        self.available_angle = min(360,available_angle)
        self.angle_step = angle_step

        self.range = sensor_range

        self.sensed_points = list()        
        self.return_all_points = return_all_points

        self.points_number = points_number
        
        self.sensed_points = list()
        
        self.data_shape = int(self.available_angle/self.angle_step)        
        if self.return_all_points:
             self.data_shape=self.data_shape*points_number  
    
    def __len__(self):
        return self.data_shape
    
    def scan(self, env):
        """строит поля точек лидара.
           Входные параметры:
           env (Game environment):
               среда, в которой осуществляется сканирование;
            return_all_points (bool):
                если False, возвращает только крайние точки лучей лидара, иначе -- все точки;
            discretization_rate (int):
                промежуток (в пикселях), через который рассматриваются точки луча лидара.
            
            Возвращает:
            sensed_points (list):
                список точек, которые отследил лидар.
            """

        # Если на нужной дистанции нет ни одного объекта - просто рисуем крайние точки, иначе нужно будет идти сложным путём
        objects_in_range = list()
        
        return_all_points = self.return_all_points
        env_range = self.range * env.PIXELS_TO_METER

        for cur_object in env.game_object_list:
            if cur_object is env.follower:
                continue
                
            if cur_object.blocks_vision:
                if distance_to_rect(self.position,cur_object) <= env_range+(3*env.PIXELS_TO_METER):
                    objects_in_range.append(cur_object)
        

        # Далее определить, в какой стороне находится объект из списка, и если он входит в область лидара, ставить точку как надо
        # иначе -- просто ставим точку на максимуме
        border_angle = int(self.available_angle/2)

        x1 = self.position[0]
        y1 = self.position[1]

        sensed_points = list()
        angles = list()
        
        cur_angle_diff = 0
        
        angles.append(self.direction)
        
        while cur_angle_diff < border_angle:
            
            cur_angle_diff += self.angle_step

            angles.append(angle_correction(self.direction+cur_angle_diff))
            angles.append(angle_correction(self.direction-cur_angle_diff))
        
        sensed_points = list()
        
        for angle in angles: 

            x2,y2 = (x1 + env_range * cos(radians(angle)), y1 - env_range * sin(radians(angle)))

            point_to_add = None
            object_in_sight = False
            
            for i in range(0,self.points_number):
                u = i/self.points_number
                cur_point = ((x2*u + x1 * (1-u)),(y2*u + y1 * (1-u)))
                
                if return_all_points:
                    sensed_points.append(cur_point)
                
                for cur_object in objects_in_range:
                    if cur_object.rectangle.collidepoint(cur_point):
                        point_to_add = np.array(cur_point,dtype=np.float32)
                        object_in_sight = True
                        break
                
                if object_in_sight:
                    break

            if point_to_add is None:
                point_to_add = np.array((x2,y2),dtype=np.float32)
            
            if not return_all_points:
                sensed_points.append(point_to_add)
        
        self.sensed_points = sensed_points
        return sensed_points


    def show(self,env):
        for cur_point in self.sensed_points:
            pygame.draw.circle(env.gameDisplay, env.colours["pink"], cur_point, 3)
        

#     @staticmethod
#     def _add_noise(val,variance):
#         return max(np.random.normal(val,variance), 0)

def angle_correction(angle):
    if angle>=360:
        return angle-360
    
    if angle<0:
        return 360+angle
    
    return angle

def angle_to_point(cur_point,target_point):
    relative_position = target_point - cur_point
        
        
    if relative_position[0]>0:
        res_angle = degrees(atan(relative_position[1]/relative_position[0]))
    elif relative_position[0]<0:
        res_angle = degrees(atan(relative_position[1]/relative_position[0]))+180
    else:
        res_angle = 0
        
    return angle_correction(res_angle)

            
def distance_to_rect(cur_point, object2):
    
    min_distance = np.inf
    for second_point in [object2.rectangle.topleft, 
                         object2.rectangle.bottomleft, 
                         object2.rectangle.topright, 
                         object2.rectangle.bottomright,
                         object2.rectangle.midtop, 
                         object2.rectangle.midleft, 
                         object2.rectangle.midbottom, 
                         object2.rectangle.midright]:
        
        cur_distance = distance.euclidean(cur_point,second_point)
        if cur_distance < min_distance:
            min_distance = cur_distance
    
    return min_distance