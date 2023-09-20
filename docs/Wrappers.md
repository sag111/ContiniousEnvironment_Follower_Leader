## Wrappers 
Wrapper classes around the environment class for pre/post-processing of observations and actions during the interaction 
of the environment and the control algorithm. Custom wrappers are implemented in the 
[wrappers](../src/continuous_grid_arctic/utils/wrappers.py) module. It is important that when adding new sensors, 
you need to add their processing to the wrapper used.
- [ContinuousObserveModifier_v0](../src/continuous_grid_arctic/utils/wrappers.py)
- [ContinuousObserveModifierPrev](../src/continuous_grid_arctic/utils/wrappers.py) to accumulate previous values of 
two upgraded sensors (1) Ray sensor with 12 rays for the corridor and obstacles; 2) Ray sensor for obstacles with 30 
(variably) rays
- [ContinuousObserveModifier_lidarMap2d](../src/continuous_grid_arctic/utils/wrappers.py) converts the lidar outputs 
into a 2D image, which displays: obstacles, leader position, safe zone on the route
- [ContinuousObserveModifier_lidarMap2d_v2](../src/continuous_grid_arctic/utils/wrappers.py) same as the previous one, 
but the safe zone is drawn differently

## Deprecated
- [MyFrameStack](../src/continuous_grid_arctic/utils/wrappers.py) for accumulating observations and using information 
not only from the current step, but also from previous ones. It has not been updated for a long time, it is now used 
instead [ContinuousObserveModifierPrev](../src/continuous_grid_arctic/utils/wrappers.py)
- [LeaderTrajectory_v0](../src/continuous_grid_arctic/utils/wrappers.py) needed only to check backward compatibility 
with experiments launched on commit 86211bf4a3b0406e23bc561c00e1ea975c20f90b
