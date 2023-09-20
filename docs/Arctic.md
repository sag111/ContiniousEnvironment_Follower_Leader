## Description
A description of the integration of a robot motion control system for the "following the leader" task in a 3D 
environment is presented here. 
Software integration is in [folder](../src/arctic_gym).
It presents the main functionality for integrating this model into the Gazebo environment with a mobile robot, which 
you can use in your own project.

A description of the 3D environment configuration is presented in the file [config.conf](../config/config.conf).

For register new environment in Gym library edit [\_\_init\_\_.py](../src/arctic_gym/__init__.py) file

## Main modules
- [arctic_env.py](../src/arctic_gym/arctic_env/arctic_env.py) integration of a gym-like environment for interaction
with ROS. File containing all the scripts for solving the following problem: obtaining lidar and camera data, r
ecognizing objects in an image, etc

- [robot_gazebo_env.py](../src/arctic_gym/base_arctic_env/robot_gazebo_env.py) default class of gym environment

- [n_routes.py](../src/arctic_gym/eval/n_routes.py) evaluate current RL model on specified routes described in 
[experiment.conf](../config/experiment.conf)

- [executor.py](../src/arctic_gym/gazebo_utils/executor.py) contains class for execute "following the leader" task and 
launches the control system and solves the problem of following in the simulation world. By running this file, the model
is launched and basic information is obtained from the environment. There is also auxiliary functionality in the form 
of a robot security system.

- [gazebo_tracker.py](../src/arctic_gym/gazebo_utils/gazebo_tracker.py) contains classes that implement sensors adapted 
for a 3D environment.

- [arctic_server.py](../src/arctic_gym/server/arctic_server.py) service for launching an RL model using the ray\[rllib\]
library. In the [config](../src/arctic_gym/server/config) folder there are RL model configuration files, in the 
[checkpoints](../src/arctic_gym/server/checkpoints) folder there are saved training checkpoints in a 2D environment.

- [publishers.py](../src/arctic_gym/topic/publishers.py) and [subscribers.py](../src/arctic_gym/topic/subscribers.py) 
contain classes for interacting with ROS Topic Subscriber/Publisher methods


## Features

To use this robot control unit, the user must complete the initial setup. The software module using the RL following 
model is configured to use the ROS Gazebo simulation world, which implements a specific robot with lidar and a 
rotating camera.
To start the demonstration, the user needs to create his own simulation world with a mobile wheeled robot, which will 
have a camera and lidar on board. The user must first have:
- World of ROS Gazebo (as an example)
- Robot with camera (if global coordinates are not used) and lidar
- Object detection (if global coordinates are not used)

The descriptions for the corresponding modules provide a description of the functions that need to be configured to 
interact with the user's own environment.  


### Safety system features
- Stopping the leader if it is too far when following;
- Emergency stop of the agent, in case of being close to the leader;
- Starting the leader search mode in case of long-term loss of sight:
  - In the event of a long-term loss of the leader, the agent begins to search by starting the rotation of the camera. 
  Function for rotating the camera _rotate_the_camera. Stops if a leader is found.
  - If there are points in the history of the leader's route, the agent, after losing it from sight, will continue to 
  move along the route and then stop. The agent will rotate the camera and stand still until it finds the leader.
  - If no actions more than 30 seconds, the task will complete automatically. 


## ArcticEnv

В файле [arctic_env.py](../src/arctic_gym/arctic_env/arctic_env.py) contains the basic structure in accordance with
[follow_the_leader_continuous_env](../src/continuous_grid_arctic/follow_the_leader_continuous_env.py), adapted to 3D 
environment. This class implements various functions for interacting with a robot, camera, lidar, route calculation, 
and more.

Main functions of the ArcticEnv
- **reset** - function of initial environment conditions

- **step** - function of processing one simulation step, processing information and calculating the required values

- **calculate_points_angles_objects** - function for calculating angles using bounding box values.

- **_get_ssd_lead_information** - function sends an image from a camera to a Flask server with object detection and 
receives object detection results. self.sub.get_from_follower_image refers to the ROS topic for an image from the 
camera. The user needs to pass his own image to this function, which is sent to his own object detection model via a 
post request. The method is used in reset and step, in those places the user needs to change the image feed or register
his own ROS topic in [config.conf](../config/config.conf) file.

- **get_lidar_points** - function of obtaining a point cloud from a lidar. The user needs to correct it to use their own 
lidar. It is necessary to register a topic to connect to the ROS Gazebo lidar in [config.conf](../config/config.conf) 
file.
 
- **_get_obs_points** - function processes the lidar point cloud for obstacle detection. Initially, the function filters 
surface points using the CSF (Cloth Simulation Filter) method, leaving only obstacle points. Next, it normalizes them 
relative to the local position of the agent. Afterwards, the resulting list is projected onto a 2D plane, the 
discreteness is reduced and the coordinate values of the remaining points are rounded. The list is added with second 
neighboring imaginary points in the immediate vicinity to form segments, which are further verified by the lidar 
features of the neural network model.

- **calculate_length_to_leader** - function determines the distance to the leader based on processing the result of 
detecting objects in the image and the lidar point cloud. Based on the received bounding boxes, they are compared with 
lidar points using the result of the calculate_points_angles_objects function. As a result, only lidar points are 
selected from the entire lidar point cloud using BB and angles from calculate_points_angles_objects. Next, based on the 
information received, the nearest point is taken, and based on it, the distance to the leader is calculated.

- **_get_camera_lead_info** - function determines the angle of deviation of the leader relative to the agent based on 
information from the camera and the distance to the leader

- **_get_delta_position** - function for determining movement in one iteration. The movement of the slave along the x 
and y coordinates in one step is determined.

- **_is_done** - function for checking task completion statuses and emergency situations.


## Sensors for 3D environment

[gazebo_tracker.py](../src/arctic_gym/gazebo_utils/gazebo_tracker.py) implemented sensors from a 2D environment, 
adapted to a 3D environment. There are currently 2 sensors in the file:
- GazeboLeaderPositionsTracker_v2
- GazeboCorridor_Prev_lasers_v2

**GazeboLeaderPositionsTracker_v2** implements tracking of the leader and the formation of its route history, using 
information about the position of the leader, the agent (working in local coordinates always x, y = [0, 0]), and 
movement values (delta_x and delta_y). As a result of the sensor’s operation, the history of the leader’s route is 
formed, as well as a safe zone, which builds the left and right walls. Features of the "ray sensor" are based on 
information about the distance to a given safe zone.

**GazeboLeaderPositionsCorridorLasers** implements radar features adapted to the 3D environment. As input, the class 
receives information about the position of the leader, the agent (working in local coordinates always x, y = [0, 0]), 
the history values of the leader’s route, the safe zone, and a list of obstacle points projected onto a 2D plane. 
The output generates normalized input features for the RL model.

Example parameters for GazeboLeaderPositionsTracker_v2:
- saving_period=5
- corridor_width=2
- corridor_length=25

Example parameters for Ray Sensor 1 (GazeboCorridor_Prev_lasers_v2)
- react_to_green_zone=True,
- react_to_safe_corridor=True,
- react_to_obstacles=True,
- lasers_count=12,
- laser_length=10,
- max_prev_obs=10,
- pad_sectors=False)

Example parameters for Ray Sensor 2 (GazeboCorridor_Prev_lasers_v2)
- react_to_green_zone=False,
- react_to_safe_corridor=False,
- react_to_obstacles=True,
- lasers_count=24,
- laser_length=15,
- max_prev_obs=10,
- pad_sectors=False)

The user can create their own sensors and input features for the model, in which case they must be implemented in 
[gazebo_tracker.py](../src/arctic_gym/gazebo_utils/gazebo_tracker.py) as a new class.

## Rl model service

[arctic_server.py](../src/arctic_gym/server/arctic_server.py) contains parameters for using the neural network tracking 
model. The user needs to configure all these parameters to suit his own sensors:

- Path to the file with model configurations
```
CONFIG_PATH = SERVER_PATH.joinpath("config/3c1bc/params.json").__str__()
CHECKPOINT = SERVER_PATH.joinpath("checkpoints/3c1bc/checkpoint_000040/checkpoint-40").__str__()
```

- Number of input features
```
sensor_config = {
    'lasers_sectors_numbers': 36
}
```
- Observation space
```
observation_space = Box(
    np.zeros((10, sensor_config['lasers_sectors_numbers']), dtype=np.float32),
    np.ones((10, sensor_config['lasers_sectors_numbers']), dtype=np.float32)
)
```
- Config modification
```
config.update(
    {
        "env": None,
        "env_config": None,
        "observation_space": observation_space,
        "action_space": action_space,
        "input": _input,
        "num_workers": 1,
        "input_evaluation": [],
        "num_gpus": 1,
        "log_level": 'DEBUG',
        "explore": False,
    }
)
```
