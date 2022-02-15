import gym
from collections import deque
import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box

class MyFrameStack(ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    .. note::
        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
    .. note::
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.
    Example::
        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally
    """

    def __init__(self, env, framestack, lz4_compress=False):
        super().__init__(env)
        self.framestack = framestack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=framestack)

        low = np.tile(self.observation_space.low[...], framestack)
        high = np.tile(
            self.observation_space.high[...], framestack
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self):
        assert len(self.frames) == self.framestack, (len(self.frames), self.framestack)
        observes = np.concatenate(self.frames)

        return observes
        #return gym.wrappers.frame_stack.LazyFrames(observes, self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.framestack)]
        return self.observation()


class ContinuousObserveModifier_v0(ObservationWrapper):
    
    def __init__(self, env, lz4_compress=False):
        super().__init__(env)
        self.prev_obs = None
        self.max_diag = np.sqrt(np.power(self.DISPLAY_WIDTH,2)+np.power(self.DISPLAY_HEIGHT, 2))
        self.observation_space = Box(np.array([-1,-1,
                                      -1,
                                      -1,
                                      -1,
                                      -1,-1,
                                      -1,
                                      -1,
                                      -1,
                                      -1,-1,
                                      -1,
                                      -1,
                                      -1,
                                      -1,-1
                                      ], dtype=np.float32),
                             np.array([1,1,
                                      1,
                                      1,
                                      1,
                                      1, 1,
                                      1,
                                      1,
                                      1,
                                      1, 1,
                                      1,
                                      1,
                                      1,
                                      1, 1
                                      ], dtype=np.float32
                                      ))

    def observation(self, obs):
        # change leader absolute pos, speed, direction to relative
        relativePositions = obs[0:4] - obs[5:9]
        distance = np.linalg.norm(relativePositions[:2])
        #distanceFromBorders = [distance-(obs[-3]/self.max_diag ), (obs[-2]/self.max_diag) - distance]
        distanceFromBorders = [distance-obs[-3], obs[-2] - distance]
        obs = obs[:-3]
        
        
        if self.prev_obs is None:
            self.prev_obs = obs        
        obs_modified = np.concatenate([obs, relativePositions, [distance], distanceFromBorders])
        
        obs_modified[0] -= self.prev_obs[0]
        obs_modified[1] -= self.prev_obs[1]
        obs_modified[3] /= 360
        obs_modified[5] -= self.prev_obs[5]
        obs_modified[6] -= self.prev_obs[6]
        obs_modified[8] /= 360
        obs_modified[10] = np.clip(obs_modified[10] / (self.max_distance*2), -1, 1)
        obs_modified[11] = np.clip(obs_modified[11] / (self.max_distance*2), -1, 1)
        obs_modified[13] /= 360
        obs_modified[14] = np.clip(obs_modified[14] / (self.max_distance*2), -1, 1)
        obs_modified[15] = np.clip(obs_modified[15] / (self.max_distance*2), -1, 1)
        obs_modified[16] = np.clip(obs_modified[16] / (self.max_distance*2), -1, 1)
        self.prev_obs = obs
        #print("OBSS", obs)
        return obs_modified#np.clip(obs, -1, 1)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.prev_obs = None
        return self.observation(observation)