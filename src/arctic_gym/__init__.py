import gym

from gym.envs.registration import register
from ray.tune import register_env


def arctic_env_maker(config):
    name = config["name"]
    env = gym.make(name, **config)

    return env


register(
    id='ArcticRobotDebug-v1',
    entry_point='src.arctic_gym.arctic_env.env_debugger:DebugEnv'
)

register_env(
    'ArcticRobotDebug-v1',
    arctic_env_maker
)

register(
    id='ArcticRobot-v1',
    entry_point='src.arctic_gym.arctic_env.arctic_env:ArcticEnv'
)

register_env(
    'ArcticRobot-v1',
    arctic_env_maker
)
