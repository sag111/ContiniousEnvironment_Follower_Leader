import numpy as np
from scipy.spatial import distance
from math import pi, degrees, radians, cos, sin, atan, acos, asin, sqrt


def angle_correction(angle):
    if angle >= 360:
        return angle - 360

    if angle < 0:
        return 360 + angle

    return angle


def angle_to_point(cur_point, target_point):
    relative_position = target_point - cur_point

    if relative_position[0] > 0:
        res_angle = degrees(atan(relative_position[1] / relative_position[0]))
    elif relative_position[0] < 0:
        res_angle = degrees(atan(relative_position[1] / relative_position[0])) + 180
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

        cur_distance = distance.euclidean(cur_point, second_point)
        if cur_distance < min_distance:
            min_distance = cur_distance

    return min_distance


def rotateVector(vec, angle):
    """
    Поворачиваем вектора на угол (в градусах)
    """
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(rot, vec)


def calculateAngle(v, w):
    """
    углы между массивом точек v и точкой w
    :param v:
    :param w:
    :return:
    """
    return np.arccos(v.dot(w) / (np.linalg.norm(v, axis=1) * np.linalg.norm(w)))

def move_to_the_point(direction, 
                      position, 
                      next_point):
    """Функция автоматического управления движением к точке"""
    desirable_angle = int(angle_to_point(position, next_point))

    cur_direction = int(direction)

    if desirable_angle - cur_direction > 0:
        if desirable_angle - cur_direction > 180:
            delta_turn = cur_direction + (360 - desirable_angle)
            new_rotation_direction = -1
        else:
            delta_turn = desirable_angle - cur_direction
            new_rotation_direction = 1

    elif desirable_angle - cur_direction < 0:
        if cur_direction - desirable_angle > 180:
            new_rotation_direction = 1
            delta_turn = (360 - cur_direction) + desirable_angle
        else:
            new_rotation_direction = -1
            delta_turn = cur_direction - desirable_angle

    else:
        new_rotation_direction = 0
        delta_turn = 0

    v = distance.euclidean(position, next_point)
    w = delta_turn * new_rotation_direction
    return np.array((v,w/10))

def areDotsOnLeft(line, dots):
        """
        Определяем, лежат ли точки dots слева от прямой line
        line: ndarray [[x1, y1], [x2,y2]]
        dots: ndarray (points, coordinates)
        """
        # D = (x2 - x1) * (yp - y1) - (xp - x1) * (y2 - y1)
        d = (line[1, 0] - line[0, 0]) * (dots[:, 1] - line[0, 1]) - (dots[:, 0] - line[0, 0]) * (
                    line[1, 1] - line[0, 1])
        return d > 0.01