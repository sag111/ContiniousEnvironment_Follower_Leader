def angle_correction(angle):
    if angle >= 360:
        return angle - 360

    if angle < 0:
        return 360 + angle

    return angle
