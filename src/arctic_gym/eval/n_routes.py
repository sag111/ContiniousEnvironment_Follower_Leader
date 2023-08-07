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

now = datetime.now()

exc = Executor(config)

collects = []
for pts in experiment["easy"]:

    start = pts[:3]
    finish = pts[3:]

    exc.setup_position(start, finish)

    time.sleep(3)

    meta = exc.follow(finish)

    collects.append(meta)

    evaluation = pd.DataFrame(collects, columns=["meta", "point_a", "point_b", "target_path", "follower_path"])
    csv_path = project_path.joinpath("data/processed")
    csv_path.mkdir(parents=True, exist_ok=True)

    evaluation.to_csv(csv_path.joinpath(f"{now.strftime('%Y-%m-%d|%H:%M')}_gazebo_eval_easy.csv"), sep=';', index=False)

collects = []
for pts in experiment["hard"]:

    start = pts[:3]
    finish = pts[3:]

    exc.setup_position(start, finish)

    time.sleep(3)

    meta = exc.follow(finish)

    collects.append(meta)

    evaluation = pd.DataFrame(collects, columns=["meta", "point_a", "point_b", "target_path", "follower_path"])
    csv_path = project_path.joinpath("data/processed")
    csv_path.mkdir(parents=True, exist_ok=True)

    evaluation.to_csv(csv_path.joinpath(f"{now.strftime('%Y-%m-%d|%H:%M')}_gazebo_eval_hard.csv"), sep=';', index=False)