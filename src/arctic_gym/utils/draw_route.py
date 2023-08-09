import cv2
import re
import pandas as pd

from pathlib import Path


def xy_to_pix(coords: list) -> list:
    """
    Преобразует координаты точки из Gazebo в соответствующие пиксели на изображении карты
    :param coords: список координат
    :return:
        список пикселей
    """
    x = int(2 * (coords[0] + 80))
    y = int(2 * (80 - coords[1]))

    return [x, y]


def convert_path(p: str) -> list:
    """
    Преобразует информацию о пути из Gazebo в список двумерных координат
    :param p: информация из топика Gazebo nav_msgs.msg.Path
    :return:
        Список координат пути
    """
    # Шаблон для извлечения строк Path, которые содержат координаты x, y
    pattern = re.compile(r'position: \n(.*)\n(.*)')
    route = []
    for match in pattern.findall(p):
        # Преобразование извлеченных строк в float
        xy = list(map(lambda x: float(x.strip().split(':')[-1].strip()), match))
        route.append(xy)

    return route


project_path = Path(__file__).resolve().parents[3]

map_path = str(project_path.joinpath('data/map/landscape.bmp'))

route_path1 = project_path.joinpath('data/processed/no_dynamics.csv')
route_path2 = project_path.joinpath('data/processed/dynamics.csv')
save_dir = 'diff'

img = cv2.imread(map_path)
data_no_dyn = pd.read_csv(route_path1, sep=';')
data_dyn = pd.read_csv(route_path2, sep=';')

for i, target in enumerate(data_dyn['target_path']):
    image = img.copy()
    # путь ведущего
    # for point in convert_path(target):
    #     pixels = xy_to_pix(point)
    #     image = cv2.circle(image, pixels, radius=0, color=(0, 0, 255), thickness=-1)

    # путь робота
    for point in convert_path(data_dyn['follower_path'][i]):
        pixels = xy_to_pix(point)
        image = cv2.circle(image, pixels, radius=0, color=(0, 255, 0), thickness=-1)

    for point in convert_path(data_no_dyn['follower_path'][i]):
        pixels = xy_to_pix(point)
        image = cv2.circle(image, pixels, radius=0, color=(0, 0, 255), thickness=-1)

    # точка старта
    start_point = eval(data_dyn['point_a'][i])
    start_px = xy_to_pix(start_point)
    image = cv2.circle(image, start_px, radius=5, color=(255, 0, 0), thickness=-1)

    # точка финиша
    finish_point = eval(data_dyn['point_b'][i])
    finish_px = xy_to_pix(finish_point)
    image = cv2.circle(image, finish_px, radius=5, color=(255, 0, 255), thickness=-1)

    # meta = eval(data_dyn['meta'][i])['mission_status']

    # if meta == 'success':
    save_path = project_path.joinpath('data/{}'.format(save_dir))
    save_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path.joinpath('route{}_diff.bmp'.format(i+1))), image)
    # else:
    #     save_path = project_path.joinpath('data/{}/{}'.format(save_dir, 'fail'))
    #     save_path.mkdir(parents=True, exist_ok=True)
    #     cv2.imwrite(str(save_path.joinpath('route{}_{}.bmp'.format(i+1, 'fail'))), image)
