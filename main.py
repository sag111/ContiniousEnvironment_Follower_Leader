import gym

import follow_the_leader_continuous_env

from gym.envs.registration import register as gym_register
from gym.wrappers import flatten_observation
from gym.spaces import Box

from stable_baselines3 import PPO
from time import sleep
        
if __name__=="__main__":
    
    manual_handling = False
    
    if manual_handling:
        
        env = gym.make("Test-Cont-Env-Manual-v0")
        
        for _ in range(5000):
            obs, rewards, dones, info = env.step((0,0)) 
            if dones:
                break
            env.render()
        env.close()
    else:
    
        env = gym.make("Test-Cont-Env-Auto-v0")

        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)


        print("Обучение закончено")
        sleep(10)
        print("начинается жара")

        obs = env.reset()

        for step in range(2000):

            action, _states = model.predict(obs)
#             print(action)
            obs, rewards, dones, info = env.step(action) 
            if dones:
                break
            env.render()
#             sleep(0.1)
        env.close()
