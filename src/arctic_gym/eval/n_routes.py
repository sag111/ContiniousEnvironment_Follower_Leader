import itertools
import time
import pandas as pd

from datetime import datetime
from pathlib import Path
from pyhocon import ConfigFactory

from src.arctic_gym.gazebo_utils.executor import Executor


project_path = Path(__file__).resolve().parents[3]

config_path = project_path.joinpath('config/config.conf')
config = ConfigFactory.parse_file(config_path)

experiment_path = project_path.joinpath('config/experiment.conf')
experiment = ConfigFactory.parse_file(experiment_path)


exc = Executor(config)

collects = []
for pts in itertools.permutations(experiment['start_point'], 2):

    exc.setup_position(pts[0], pts[1])

    time.sleep(2)

    meta = exc.follow(pts[1])

    collects.append(meta)


evaluation = pd.DataFrame(collects, columns=["meta", "point_a", "point_b", "target_path", "follower_path"])
csv_path = project_path.joinpath("data/processed")
csv_path.mkdir(parents=True, exist_ok=True)

now = datetime.now()
evaluation.to_csv(csv_path.joinpath(f"{now.strftime('%Y-%m-%d|%H:%M')}_gazebo_eval.csv"), sep=';', index=False)
