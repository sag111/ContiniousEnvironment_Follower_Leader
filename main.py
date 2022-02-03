import gym
import env
from gym.envs.registration import register as gym_register

        
if __name__=="__main__":

    env = gym.make("Test-Cont-Env-v0")
    env.reset()
    
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
    env.close()
