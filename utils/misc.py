import numpy as np


def angle_correction(angle):
    if angle >= 360:
        return angle - 360

    if angle < 0:
        return 360 + angle

    return angle


def rotateVector(vec, angle):
    """
    Поворачиваем вектора на угол (в градусах)
    """
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(rot, vec)


def calculateAngle(v, w):
    """

    :param v:
    :param w:
    :return:
    """
    return np.arccos(v.dot(w) / (np.linalg.norm(v, axis=1) * np.linalg.norm(w)))