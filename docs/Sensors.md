
### Sensors description
The environment contains a number of sensors for the robot to interact with the world.
Sensors are implemented in the file [sensors.py](../src/continuous_grid_arctic/utils/sensors.py)

#### Main
- **LeaderPositionsTracker_v2** - (main) sensor that allows you to read the position of the leader and, based on this 
information, stores the history of the leader’s route points of a certain length and customizable travel corridor
- LeaderCorridor_lasers - "ray sensor" with 7 rays (baseline), input features for a neural network model based on 
measuring information about the distance to obstacles and the corridor
- LeaderCorridor_lasers_v2 - "ray sensor" with an adjustable number of rays (12 in the basic configuration) reacting 
to the corridor and obstacles
- LeaderObstacles_lasers - "ray sensor" with an adjustable number of rays, reacting only to obstacles (two types: 
static and dynamic)
- **LeaderCorridor_Prev_lasers_v2** - (main) "ray sensor" with the ability to store information about obstacles in 
previous steps. This information allows you to store information about the values of the rays at previous steps, taking 
into account the recalculation of these values relative to the current position of the robot at each new step. Has 
customizable ray values and history lengths. Determines the closest distance to an obstacle or travel corridor.
- **LaserPrevSensor** - (main) "ray sensor", similar to LeaderCorridor_Prev_lasers_v2. However, it only determines the 
closest distance to obstacles.

- **LeaderCorridor_Prev_lasers_v2_compas** - a new “ray sensor”, similar to LeaderCorridor_Prev_lasers_v2, but with the 
ability to determine directions in the generated features, the original vector increases in length by 4 times, and is 
broken by zeros in unused sides. The OBSERVATION example consists of concatenated feature vectors, which are formed as 
follows front = [1, 1, 0, 0, 0, 0, 0, 0], right = [0, 0, 1, 1, 0, 0, 0, 0 ], back = [0, 0, 0, 0, 1, 1, 0, 0], 
left = [0, 0, 0, 0, 1, 1]. RES_OBS = np.concatenate((front, right, behind, left), axis=None)
- **LaserPrevSensor_compas** - a new “ray sensor”, similar in principle to LaserPrevSensor, with direction indication 
as LeaderCorridor_Prev_lasers_v2_compas.

**IMPORTANT:**
The LeaderCorridor_Prev_lasers_v2_compas and LaserPrevSensor_compas sensors are subject to similar rules as the 
LeaderCorridor_Prev_lasers_v2 and LaserPrevSensor

#### Deprecated:
- LaserSensor - "the leader features"
- LeaderPositionsTracker - sensor for generating a history of leader route points
- LeaderTrackDetector_vector - sensor for determining the history of the leader's waypoints
- LeaderTrackDetector_radar - sensor to determine the sector in which the leader’s route history is located
- GreenBoxBorderSensor - safety zone detection sensor based on lidar. Very slow, unoptimized implementation
- Leader_Dyn_Obstacles_lasers - "ray sensor" to determine the distance only to dynamic obstacles
- FollowerInfo - sensor for generating input features from information about the linear and angular speed of the robot

#### Using LeaderCorridor_Prev_lasers_v2 and LaserPrevSensor:
These sensors accumulate the history of obstacle coordinates and normalize them relative to the current position of 
the agent. When using these sensors, you must set two parameters in the configuration:
- parameter use_prev_obs is bool.
- parameter max_prev_obs takes values of the number of accumulated steps in history. Default is 5

#### Add custom sensor
To add your own sensor, you need to write a class in the [sensors.py](../src/continuous_grid_arctic/utils/sensors.py)
and add the name of this sensor to the SENSOR_NAME_TO_CLASS variable at the end of the file. Also, you need to register 
the new sensor in the file [wrappers.py](../src/continuous_grid_arctic/utils/wrappers.py)

There are two options:
1. By analogy, add the resulting sensor to the ContinuousObserveModifier_v0 class.
2. Write your own class similar to ContinuousObserveModifier_v0.

To use a sensor, you need to specify the name and parameters of the sensor in the follower_sensors variable in the 
configuration.

Example of using the sensor:

```
 follower_sensors={
     'LeaderPositionsTracker_v2': {
         'sensor_name': 'LeaderPositionsTracker_v2',
         'eat_close_points': True,
         'saving_period': 8,
         'start_corridor_behind_follower':True
     },
     "LeaderCorridor_Prev_lasers_v2": {
            'sensor_name': 'LeaderCorridor_Prev_lasers_v2',
            "react_to_obstacles": True,
            "front_lasers_count": 2,
            "back_lasers_count": 2,
            "react_to_safe_corridor": True,
            "react_to_green_zone": True,
            "laser_length": 150
     }
 }
```

Also, if necessary, you can write additional logic for using the sensor in 
[the agent class](../src/continuous_grid_arctic/utils/classes.py). For example, the LeaderPositionsTracker or 
LeaderPositionsTracker_v2 sensors must work before the sensors that use their readings. And some sensors may require 
input from the scan() function, in addition to a link to the environment, and also the readings of previous sensors.


### Examples of adding main sensors
#### LeaderTrackDetector_vector
Example of using radar features:
<p align="center">
<img src="../src/continuous_grid_arctic/figures/LeaderTrackDetector_vector_2.jpg" width="500">
</p>

#### LaserSensor
Example of using lidar features:
<p align="center">
<img src="../src/continuous_grid_arctic/figures/LaserSensor.jpg" width="500">
</p>

#### LeaderCorridor_lasers
Example of using the features of a "ray sensor" in a main configuration with 7 rays. 
Reacts to the corridor and obstacles:
<p align="center">
<img src="../src/continuous_grid_arctic/figures/LeaderCorridor_lasers.jpg" width="500">
</p>

#### LeaderCorridor_lasers_v2
Example of using “ray sensor” features with a customizable number of rays (default 12) with response to the corridor 
and obstacles:
<p align="center">
<img src="../src/continuous_grid_arctic/figures/LeaderCorridor_lasers_v2.jpg" width="500">
</p>

#### LeaderObstacles_lasers
Example of using the attributes of a “ray sensor” that reacts only to obstacles (30 rays by default):
<p align="center">
<img src="../src/continuous_grid_arctic/figures/LeaderObstacles_lasers.jpg" width="500">
</p>

The **LeaderCorridor_Prev_lasers_v2** and **LaserPrevSensor** sensors are similar to **LeaderCorridor_lasers_v2** and 
**LeaderObstacles_lasers** respectively. Their differences are that they save the history of obstacle points and 
normalize the values at each step relative to the current position of the agent.