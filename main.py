import gym

import follow_the_leader_continuous_env

from gym.envs.registration import register as gym_register
from gym.wrappers import flatten_observation
from gym.spaces import Box

from stable_baselines3 import PPO
from time import sleep

import argparse as ag
        
if __name__=="__main__":
    
    parser = ag.ArgumentParser()
    
    parser.add_argument('--manual', 
                        action='store_true', 
                        help="Если указано, управление осуществляется стрелочками вручную, иначе на основе поданных моделью действий")
    
    parser.add_argument('--n_steps',
                        type=int,
                        default=5000,
                        help="Число шагов, в течение которых работает проверочная симуляция")
    
    parser.add_argument('--training_steps',
                    type=int,
                    default=10000,
                    help="Число шагов, в течение которых происходит обучение модели (только для режима управления автоматом)")
    
#     parser.add_argument('--n_sim',
#                         type=int,
#                         default=1,
#                         help="Число симуляций")
    
    args = parser.parse_args()
    
    manual_handling = args.manual #False
    
    if manual_handling:
        
        env = gym.make("Test-Cont-Env-Manual-v0")
        action = (0,0)
        
        for _ in range(args.n_steps):
            obs, rewards, dones, info = env.step((0,0)) 
            if dones:
                break
            env.render()
        env.close()
        
    else:
        env = gym.make("Test-Cont-Env-Auto-v0")

        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=args.training_steps)

        print("Обучение закончено")
        sleep(10)
        print("начинается управление")

        obs = env.reset()
        for step in range(args.n_steps):

            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action) 
            if dones:
                break
            env.render()
        env.close()
