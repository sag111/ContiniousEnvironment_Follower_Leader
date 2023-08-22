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
    parser.add_argument("--mode",
                        type=str,
                        default="manual",
                        help="режим работы эксперимента. manual - ручной, rl - обучение алгоритма, base - использование базового алгоритма")
    
    parser.add_argument('--n_steps',
                        type=int,
                        default=20000,
                        help="Число шагов, в течение которых работает проверочная симуляция")
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help="Рандом сид для инициализации среды (от него зависит расположение случайно генерируемых препятствий)")
    parser.add_argument('--log_results',
                        action="store_true",
                        help="Надо ли сохранять результаты эпизодов в файл")
    parser.add_argument('--hardcore',
                        action="store_true",
                        help="Если выбран ручной режим и включен режим hardcore, то будут отображаться только показатели сенсоров")
    
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

    if args.mode=="manual":
        manual_handling=True
    
    if manual_handling:
        

        if args.hardcore:
            env = gym.make("Test-Cont-Env-Manual-hardcore")
        else:
            env = gym.make("Test-Cont-Env-Manual-v0")
        # сиды с кривыми маршрутами: 9, 33
        # лидер сталкивается с препятствием: 21,22, 32, 33
        if args.seed is not None:
            env.seed(args.seed)
        env.reset()

        action = (0, 0)
        
        for f_i in range(args.n_steps):
            obs, rewards, dones, info = env.step((0,0)) 
            if dones:
                print(
                    "Эпизод закончен: итоговая награда {}, пройдено кадров {} статус миссии {}, статус ведущего {}, "
                    "статус ведомого {}".format(env.overall_reward, env.step_count, info["mission_status"],
                                                info["leader_status"], info["agent_status"]))
                if args.log_results:
                    if not os.path.exists("runs_results.csv"):
                        with open("runs_results.csv", "w") as f:
                            f.write("seed;reward;frames;mission_status;leader_status;agent_status\n")
                    with open("runs_results.csv", "a") as f:
                        f.write("{};{};{};{};{};{}\n".format(args.seed, env.overall_reward, env.step_count, info["mission_status"],
                                                info["leader_status"], info["agent_status"]))
                break
            env.render()
        sleep(1)
        #env.close()
        
    
    
    if args.mode=="base":
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
        
        
    if args.mode=="rl":

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


        
