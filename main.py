import gym
import follow_the_leader_continuous_env
from gym.envs.registration import register as gym_register

        
if __name__=="__main__":

    env = gym.make("Test-Cont-Env-v0")
    env.reset()
    
    for _ in range(10000):
        env.render()
        env.step((2.,0)) 
    env.close()
