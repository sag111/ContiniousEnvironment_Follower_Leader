import pygame
from math import pi, degrees, radians, cos, sin, atan, acos, asin, sqrt
import numpy as np
from scipy.spatial import distance
try:
    from utils.misc import angle_correction, angle_to_point
    from utils.sensors import SENSOR_NAME_TO_CLASS
except:
    from continuous_grid_arctic.utils.misc import angle_correction, angle_to_point
    from continuous_grid_arctic.utils.sensors import SENSOR_NAME_TO_CLASS
    
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
            self.image = pygame.transform.scale(self.image, (width, height))
        else:
            self.height = self.image.get_height()
            self.width = self.image.get_width()

        self.start_position = np.array(start_position, dtype=np.float32)
        self.position = self.start_position

        self.rectangle = self.image.get_rect(center=self.position, width=width, height=height)

        self.blocks_vision = blocks_vision

    def put(self, position):
        self.position = np.array(position, dtype=np.float32)


class AbstractRobot(GameObject):
    def __init__(self,
                 name,
                 image=None,
                 start_position=None,
                 height=None,  # в метрах
                 width=None,  # в метрах
                 min_speed=0.,  # в метрах в секунду
                 max_speed=2.,  # в метрах в секунду
                 max_rotation_speed=60,  # в градусах
                 max_speed_change=0.5,  # в метрах в секунду
                 max_rotation_speed_change=57,  # в градусах
                 start_direction=0,  # в градусах
                 sensor=None,
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

    #         self.sensor = sensor

    def command_turn(self, desirable_rotation_speed, rotation_direction):
        """Обработка команд, влияющих на скорость угловую w"""
        # Не превышаем максимальной скорости

        self.desirable_rotation_speed = min(desirable_rotation_speed, self.max_rotation_speed)
        self.desirable_rotation_direction = rotation_direction

        if (rotation_direction == 0) and (desirable_rotation_speed != 0):
            raise ValueError("Указана скорость поворота, но направление = 0!")

    def command_forward(self, desirable_speed):
        """Обработка команд, влияющих на скорость v"""
        if desirable_speed > self.max_speed:
            desirable_speed = self.max_speed

        if desirable_speed < self.min_speed:
            desirable_speed = self.min_speed

        self.desirable_speed = desirable_speed

    def _controller_call(self):
        """Изменение скорости в зависимости от установленных желаемых скоростей на основе управления"""
        self._turn_processing()
        self._speed_processing()

    def _turn_processing(self):
        """Обработка изменения скорости поворота в такт для контроллера"""
        if self.rotation_direction == 0:
            self.rotation_direction = self.desirable_rotation_direction

        if self.rotation_direction == self.desirable_rotation_direction:
            needed_change = abs(self.rotation_speed - self.desirable_rotation_speed)
            speed_rotation_change = min((needed_change, self.max_rotation_speed_change))

            if self.desirable_rotation_speed < self.rotation_speed:
                speed_rotation_change = -1 * speed_rotation_change
        else:
            needed_change = abs(self.desirable_rotation_speed + self.rotation_speed)
            speed_rotation_change = -min((needed_change, self.max_rotation_speed_change))

        new_rotation_speed = self.rotation_speed + speed_rotation_change

        if new_rotation_speed < 0:
            self.rotation_direction = -1 * self.rotation_direction
        self.rotation_speed = abs(new_rotation_speed)

    def _speed_processing(self):
        """Обработка изменения скорости в такт для контроллера"""
        needed_change = abs(self.speed - self.desirable_speed)
        speed_change = min(self.max_speed_change, needed_change)

        if self.speed > self.desirable_speed:
            speed_change = -1 * speed_change

        self.speed = self.speed + speed_change

    def move(self):
        """Функция, которая перемещает робота с учётом установленных желаемых скоростей."""
        # скорректировали скорости
        self._controller_call()
        if self.rotation_speed != 0:

            self.direction = angle_correction(self.direction + self.rotation_direction * self.rotation_speed)
            # TODO: объединить изменение положения хитбокса и изменение размера в соответствии с поворотом
            # Rotate the original image without modifying it.
            new_image = pygame.transform.rotate(self.image, -self.direction)
            # Get a new rect with the center of the old rect.
            self.rectangle = new_image.get_rect(center=self.rectangle.center)

        movement_vec = np.array((cos(radians(self.direction)) * self.speed, sin(radians(self.direction)) * self.speed),
                                dtype=np.float32)
        self.position += movement_vec
        position_diff = self.position - self.rectangle.center
        if np.linalg.norm(position_diff) > 0:
            self.rectangle.move_ip(position_diff)

    def move_to_the_point(self, next_point, speed=None):
        """Функция автоматического управления движением к точке"""
        
        if speed is not None:
            new_speed = speed
        else:
            new_speed = distance.euclidean(self.position, next_point)
        
        desirable_angle = int(angle_to_point(self.position, next_point))

        cur_direction = int(self.direction)

        if desirable_angle - cur_direction > 0:
            if desirable_angle - cur_direction > 180:
                delta_turn = cur_direction + (360 - desirable_angle)
                new_rotation_direction = -1
            else:
                delta_turn = desirable_angle - cur_direction
                new_rotation_direction = 1

        else:
            if cur_direction - desirable_angle > 180:
                new_rotation_direction = 1
                delta_turn = (360 - cur_direction) + desirable_angle
            else:
                new_rotation_direction = -1
                delta_turn = cur_direction - desirable_angle

        self.command_turn(delta_turn, new_rotation_direction)
        self.command_forward(new_speed)

        self.move()


class RobotWithSensors(AbstractRobot):
    def __init__(self,
                 name,
                 image=None,
                 start_position=None,
                 height=None,  # в метрах
                 width=None,  # в метрах
                 min_speed=0.,  # в метрах в секунду
                 max_speed=2.,  # в метрах в секунду
                 max_rotation_speed=60,  # в градусах
                 max_speed_change=0.5,  # в метрах в секунду
                 max_rotation_speed_change=57,  # в градусах
                 start_direction=0,  # в градусах
                 blocks_vision=True,
                 sensors={},
                 **kwargs
                 ):
        super().__init__(name, image, start_position, height, width, min_speed, max_speed,
                         max_rotation_speed, max_speed_change, max_rotation_speed_change, start_direction,
                         blocks_vision)
        self.sensors = {}
        # TODO: Чтобы можно было несколько сенсоров одного типа, надо в качестве ключей использовать имена, и добавлять поле - класс
        # например если надо будет отслеживание истории позиций лидера с поеданием точек и без для разных целей
        for k in sensors:
            if k in ['LeaderTrackDetector_vector', 'LeaderTrackDetector_radar']:
                if 'LeaderPositionsTracker' not in sensors.keys() and "LeaderPositionsTracker_v2" not in sensors.keys():
                    raise ValueError("Sensor {} requires sensor LeaderPositionsTracker for tracking leader movement.".format(k))
            self.sensors[k] = SENSOR_NAME_TO_CLASS[k](self, **sensors[k])

    def use_sensors(self, env):
        sensors_observes = dict()
        if 'LeaderPositionsTracker' in self.sensors.keys():
            if self.sensors['LeaderPositionsTracker'].generate_corridor:
                leader_positions_hist, leader_corridor = self.sensors['LeaderPositionsTracker'].scan(env)
            else:
                leader_positions_hist = self.sensors['LeaderPositionsTracker'].scan(env)
        if 'LeaderPositionsTracker_v2' in self.sensors.keys():
            if self.sensors['LeaderPositionsTracker_v2'].generate_corridor:
                leader_positions_hist, leader_corridor = self.sensors['LeaderPositionsTracker_v2'].scan(env)
            else:
                leader_positions_hist = self.sensors['LeaderPositionsTracker_v2'].scan(env)
        for sensor_name, sensor in self.sensors.items():
            if sensor_name == 'LeaderPositionsTracker':
                continue
            if sensor_name in ['LeaderTrackDetector_vector', 'LeaderTrackDetector_radar']:
                sensors_observes[sensor_name] = sensor.scan(env, leader_positions_hist)
            elif sensor_name in ['LeaderCorridor_lasers', 'LeaderCorridor_lasers_v2']:
                sensors_observes[sensor_name] = sensor.scan(env, leader_corridor)
            else:
                sensors_observes[sensor_name] = sensor.scan(env)


        return sensors_observes
