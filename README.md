# continuous-grid-arctic

## 2D environment
[src/continuous_grid_arctic](src/continuous_grid_arctic) contains gym environment for "following the leader" task. 

- [Environment classes description](docs/README.md)
- [Sensors description](docs/Sensors.md)
- [Wrappers description](docs/Wrappers.md)

## 3D environment
[src/arctic_gym](src/arctic_gym) contains gym-gazebo environment "following the leader" task.

- [Gym-gazebo description](docs/Arctic.md)

## Setup
```
git clone https://github.com/sag111/continuous-grid-arctic
```
Use as package
```
pip install git+https://github.com/sag111/continuous-grid-arctic.git
```

### Anaconda env install
1. Installation using conda.yml 
``` 
conda env create -f conda.yml 
```

2. If errors occurred:
``` 
conda create -n rl -c conda-forge python-pdal=3.1.2 python=3.7; 
conda activate rl;
pip install -r requirements.txt
```

3. For 2D environment 
```
pip install pygame==2.1.2 pandas numpy
``` 

### Requirements:
- Python 3.7.12
- setuptools==66.0.0
- wheel==0.38.4
- opencv-python==4.5.4.60
- ray\[rllib\]==1.9.5
- pygame==2.1.2
- pyhocon==0.3.60
- rospkg==1.4.0
- importlib-metadata==4.13.0
- open3d==0.17.0
- torch==1.13.1
- protobuf==3.20

## 2D environment usage:
To demonstrate how the environment works in manual mode run the file main.py:
```
python run_2d.py --mode manual --seed 0 --hardcore --manual_input gamepad --log_results
```
Possible command line arguments are described in the script [run_2d.py](src/run_2d.py). In addition, you can configure 
the environment in the script follow_the_leader_continus_env.py. To do this, you need to look in run_2d.py to see which 
environment is being launched and change the parameters in follower_the_leader_continuous_env. For example, you can 
speed up the simulation by increasing the framerate or frames_per_step.

By default, one simulation lasts no more than 5000 steps (set when creating a specific environment with the max_steps 
parameter) or until the agent gets into an accident.

The notebooks folder contains two demo Jupyter notebooks.
1. [Env_demo](src/continuous_grid_arctic/notebooks/Env_demo.ipynb) contains a demo program for interacting with 
the environment
2. [Ray_train_demo](src/continuous_grid_arctic/notebooks/Ray_train_demo.ipynb) contains a demo program for training the 
agent and testing the resulting model using the ray[rllib] library

Below is a test run of one route using **LeaderPositionsTracker_v2**, **LeaderCorridor_Prev_lasers_v2**, 
**LaserPrevSensor**. The model was trained in an environment configuration with 35 static obstacles and 1 dynamic one.

<p align="center">
<img src="src/continuous_grid_arctic/figures/demo_video.gif" width="500">
</p>

## 3D environment usage:

To demonstrate the operation of the model in a 3D environment, you must read the instructions 
[Arctic.md](docs%2FArctic.md)
<p align="center">
<img src="src/arctic_gym/figures/demo_gazebo.gif" width="500">
</p>

## Configuring your own environment
To create your own environment configuration, you must complete the following steps:
1. In the follow_the_leader_continuous_env.py file, create a class that inherits the main environment 
(such as Test-Cont-Env-Auto-v0);
2. In the init method of the created class, set the necessary parameters when initializing the parent class (for a 
complete list of parameters, see the init method of the Game class);
3. Next, “register” the environment as a gym environment using gym_register, using the following template:
   ```
   id=Test-Cont-Env-<your_env_name>-v0;
   follow_the_leader_continuous_env:<name of the environment class that was created in step 1>;
   reward_threshold at will.
   ```

## Citation

Please use this bibtex if you want to cite this repository in your publications:
```
@article{selivanov2022environment,
  title={An environment emulator for training a neural network model to solve the “Following the leader” task},
  author={Selivanov, Anton and Rybka, Roman and Gryaznov, Artem and Shein, Vyacheslav and Sboev, Alexander},
  journal={Procedia Computer Science},
  volume={213},
  pages={209--216},
  year={2022},
  publisher={Elsevier}
}
```
