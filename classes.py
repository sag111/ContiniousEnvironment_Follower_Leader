import pygame
from math import pi, degrees, radians, cos, sin, atan, acos, asin, sqrt
import numpy as np
from scipy.spatial import distance

class GameObject():
    def __init__(self,
                 name,
                 image=None,
                 start_position=None,
                 height=None,
                 width=None):
        """Класс, который отражает любой игровой объект и должен наследоваться конкретизирующими классами.
        Входные параметры:
        image (pygame.image):
            изображение объекта;
        start_position (tuple(int,int)):
            стартовые координаты объекта;
        height, width (int):
            размеры объекта в пикселях.
        """
        self.name = name
        self.image = image
        
        if height and width:
            self.image = pygame.transform.scale(self.image, (width,height))
            
        self.start_position = np.array(start_position)
        self.position = self.start_position
        
        self.rectangle = self.image.get_rect(center=self.position, width=width, height=height)
        
    def put(self,position):
        self.position = np.array(position)
    
        
        

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
                 **kwargs
                 ):
        """Класс, который реализует робота."""
        # TODO: задание формы робота полигоном или абстрактной фигурой
        
        super(AbstractRobot, self).__init__(name,
                 image=image,
                 start_position=start_position,
                 height=height,
                 width=width)
        
        self.speed = 0.
        self.rotation_speed = 0.
        self.rotation_direction = 0
        # 0 -- прямо, -1 -- влево, +1 -- вправо.
        
        self.direction = start_direction # в градусах!
        
        self.desirable_rotation_speed = 0. # в радианах
        self.desirable_rotation_direction = 0.
        self.desirable_speed = 0.
        
        self.min_speed = min_speed
        self.max_speed = max_speed
        
        self.max_rotation_speed = max_rotation_speed
        
        self.max_speed_change = max_speed_change
        self.max_rotation_speed_change = max_rotation_speed_change
        
        self.width = width
        self.height = height
        
        
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
            self.direction = self.direction + self.rotation_direction*self.rotation_speed
            if self.direction > 360:
                self.direction = self.direction-360
            if self.direction < 0:
                self.direction = 360+self.direction
        
        movement_vec = np.array((cos(radians(self.direction))*self.speed, sin(radians(self.direction))*self.speed))
        self.position+=movement_vec
        
    
    
    def move_to_the_point(self, next_point):
        """Функция автоматического управления звижением к точке"""
        #TODO: сделать более хороший алгоритм следования маршруту [Слава]
        
        next_point_scaled = next_point - self.position # таким образом мы рассчитываем положение точки относительно робота
        
        
        if next_point_scaled[0]>0:
            desirable_angle = degrees(atan(next_point_scaled[1]/next_point_scaled[0]))
        elif next_point_scaled[0]<0:
            desirable_angle = degrees(atan(next_point_scaled[1]/next_point_scaled[0]))+180
        else:
            desirable_angle = 0
            
        if desirable_angle > 360:
            desirable_angle-=360
        
        if desirable_angle<0:
            desirable_angle=360+desirable_angle
        
        delta_turn = self.direction-desirable_angle
        
        if delta_turn > 0.:
            self.rotation_direction=-1
        elif delta_turn < 0.:
            self.rotation_direction=1
        else:
            self.rotation_direction=0
        
        delta_turn = abs(delta_turn)

        self.command_forward(distance.euclidean(self.position,next_point))
        self.command_turn(abs(delta_turn), self.rotation_direction)
        
        self.move()