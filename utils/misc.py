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
