import gym

import follow_the_leader_continuous_env

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
    
    parser.add_argument('--n_steps',
                        type=int,
                        default=5000,
                        help="Число шагов, в течение которых работает проверочная симуляция")
    
    parser.add_argument('--training_steps',
                    type=int,
                    default=10000,
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
    
    manual_handling = True#args.manual #False
    
    if manual_handling:
        
        env = gym.make("Test-Cont-Env-Manual-v0")
        env.reset()
        action = (0,0)
        
        for _ in range(args.n_steps):
            obs, rewards, dones, info = env.step((0,0)) 
            if dones:
                break
            env.render()
        env.close()
        
    else:
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


        
         