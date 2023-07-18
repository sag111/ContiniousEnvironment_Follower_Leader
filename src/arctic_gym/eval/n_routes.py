import rospy
import numpy as np

from pathlib import Path
from pyhocon import ConfigFactory

from src.arctic_gym import arctic_env_maker


project_path = Path(__file__).resolve().parents[3]

config_path = project_path.joinpath('config/config.conf')
config = ConfigFactory.parse_file(config_path)

experiment_path = project_path.joinpath('config/experiment.conf')
experiment = ConfigFactory.parse_file(experiment_path)

rospy.init_node("rl_client", anonymous=True)

env_config = config.rl_agent.env_config
env = arctic_env_maker(env_config)

# env.pub.teleport(model="arctic_model", point=[70, -30, 0.3], quaternion=[0, 0, 0, 1])

import itertools

# перестановки точек для теста
count = 0
for pts in itertools.permutations(experiment['start_point'], 2):
    print(pts)


# Определение угла от начальной к конечной точке для правильной расстановки ведущего
x0 = 50.0
x1 = -40.0

y0 = -41.0
y1 = 30.0

print(np.rad2deg(np.arctan2(y0 - y1, x1 - x0)))


