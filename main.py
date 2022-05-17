import pathlib
import sys
sys.path.append(pathlib.Path().resolve())
import gym

from continuous_grid_arctic.follow_the_leader_continuous_env import *

from gym.envs.registration import register as gym_register
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.spaces import Box

from time import sleep

import argparse as ag
        
if __name__ == "__main__":
    
    parser = ag.ArgumentParser()
    
    parser.add_argument('--manual', 
                        action='store_true', 
                        help="Если указано, управление осуществляется стрелочками вручную, иначе на основе поданных моделью действий")
    
    parser.add_argument("--regime",
                        type=str,
                        default="rl",
                        help="режим работы эксперимента. manual - ручной, rl - обучение алгоритма, base - использование базового алгоритма")
    
    parser.add_argument('--n_steps',
                        type=int,
                        default=15000,
                        help="Число шагов, в течение которых работает проверочная симуляция")
    
    parser.add_argument('--training_steps',
                    type=int,
                    default=50000,
                    help="Число шагов, в течение которых происходит обучение модели (только для режима управления автоматом)")
    
    parser.add_argument('--video_name',
                    type=str,
                    default="prediction.mp4",
                    help="название видео с результирующей симуляцией (только для режима управления автоматом)")
    
#     parser.add_argument('--n_sim',
#                         type=int,
#                         default=1,
#                         help="Число симуляций")
    
    args = parser.parse_args()
    
    
    manual_handling = args.manual
    
    if args.regime=="manual":
        manual_handling=True
    
    if manual_handling:
        
        env = gym.make("Test-Cont-Env-Manual-v0")
        # сиды с кривыми маршрутами: 9, 33
        # лидер сталкивается с препятствием: 21,22, 32, 33
        #env.seed(33)
        env.reset()
        action = (0,0)
        
        for _ in range(args.n_steps):
            obs, rewards, dones, info = env.step((0,0)) 
            if dones:
                break
            env.render()
        env.close()
        
    
    
    if args.regime=="base":
        from scipy.spatial import distance
        from utils.misc import angle_to_point, move_to_the_point
        import numpy as np
        
        env =  gym.make("Test-Cont-Env-Auto-Follow-with-obstacles-v0")
        env.metadata["render.modes"] = ["rgb_array"]
        
        recorder = VideoRecorder(env, "./video/{0}".format(args.video_name), enabled = True)
        
        target_point_stack = []
        
        obs = env.reset()
        target_point_stack.append(obs["leader_target_point"])
        
        for step in range(args.n_steps):
            env.render()
            recorder.capture_frame()
            
            direction = obs["numerical_features"][8]
            position = np.array((obs["numerical_features"][5], 
                                obs["numerical_features"][6]))
            
            next_point = target_point_stack[0]
            
            action = move_to_the_point(direction, 
                              position, 
                              next_point)

            
            obs, rewards, dones, info = env.step(action)
            
            if not np.array_equal(obs["leader_target_point"], target_point_stack[-1]):
                target_point_stack.append(np.array(obs["leader_target_point"]))
            
            if distance.euclidean(position,next_point) <= 20:
                _ = target_point_stack.pop(0)
            
            if dones:
                break
        
        recorder.close()
        env.close()     
        
        
    if args.regime=="rl":

        from stable_baselines3 import PPO

        env = gym.make("Test-Cont-Env-Auto-v0")

        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001)
        model.learn(total_timesteps=args.training_steps)

        print("Обучение закончено")
        sleep(10)
        print("начинается управление")

        env =  gym.make("Test-Cont-Env-Auto-v0")
        env.metadata["render.modes"] = ["rgb_array"]

        recorder = VideoRecorder(env, "./video/{0}".format(args.video_name), enabled = True)


        obs = env.reset()
        for step in range(args.n_steps):
            env.render()
            recorder.capture_frame()

            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action) 
            if dones:
                break

        recorder.close()
        env.close()


        
