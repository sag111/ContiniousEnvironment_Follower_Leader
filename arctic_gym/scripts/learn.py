#!/usr/bin/env python
import logging

import os
import rospy
import numpy as np
from src.arctic_gym import arctic_env_maker
from ray.rllib.env.policy_client import PolicyClient

from src.instruments.rosmod.subscribers import Subscribers
from src.instruments.rosmod.publishers import Publishers
from pyhocon import ConfigFactory

PATH_TO_CONFIG = os.path.join(os.getcwd(), 'CONFIG', 'config.conf')
config = ConfigFactory.parse_file(PATH_TO_CONFIG)

def _rotate_the_camera(camera_flag):

    camera_yaw_state_info = env.sub.get_camera_yaw_state()
    rot_cam = camera_yaw_state_info.process_value

    test_camera = abs(rot_cam - 3.59758)
    test_camera_1 = abs(rot_cam + 3.59758)
    rot_angl = 0.356

    if test_camera < 0.7:
        camera_flag = False

    if test_camera_1 < 0.5:
        camera_flag = True

    if camera_flag:
        rot_cam = rot_cam + rot_angl
    else:
        rot_cam = rot_cam - rot_angl

    env.pub.set_camera_yaw(rot_cam)

    return camera_flag


if __name__ == '__main__':
    rospy.init_node('arctic_gym_rl', anonymous=True, log_level=rospy.INFO)

    env_config = {"name": "ArcticRobot-v1",
                  "discrete_action": False,
                  "time_for_action": 0.3}
    env = arctic_env_maker(env_config)
    sub = Subscribers()
    pub = Publishers()

    # TODO: ip
    # client = PolicyClient("http://10.8.0.10:9900", inference_mode='remote')
    client = PolicyClient("http://192.168.1.34:9900", inference_mode='remote')

    # goal = [30.0, -50.0]
    goal = env.goal

    camera_yaw = 0
    count_lost = 0
    camera_flag = True


    while True:
        obs = env.reset()
        eid = client.start_episode(training_enabled=True)
        rewards = 0
        lead_loss = False
        speed_coeff = 1.9

        # pus_obs = np.zeros(7, dtype=np.float32)

        pus_obs = np.ones(7, dtype=np.float32)
        pus_obs[5] *= 0.65
        pus_obs[6] *= 0.65

        print(pus_obs)

        info = {
            "mission_status": "in_progress",
            "agent_status": "None",
            "leader_status": "moving"
        }

        for s in range(1000):

            action = client.get_action(eid, obs)
            # TODO : переписать систему безопасности
            if s <= 1:
                action[0] *= 0
                action[1] *= 0

            if info['leader_status'] == "moving" and count_lost >= 1:
                action[0] *= speed_coeff
                print("__________________________________________")
                print("we found a leader again")
                print("__________________________________________")
                count_lost = 0
                env.pub.move_target(goal[0], goal[1])

            if info["agent_status"] == "too_close_to_leader" and info["leader_status"] == 'moving':
                print("__________________________________________")
                print(" too close to the leader ")
                print("__________________________________________")
                action[0] *= 0

            if info['agent_status'] == 'too_far_from_leader_info' and info["leader_status"] == 'moving':
                action[0] *= speed_coeff
                print("__________________________________________")
                print(" too far to the leader info ")
                print("__________________________________________")
                count_lost += 1
                env.pub.target_cancel_action()


            if info['leader_status'] == 'None':
                print(action)
                action[0] *= speed_coeff
                # action[0] *= 0
                print("__________________________________________")
                print("we lost the leader")
                print("__________________________________________")

                if info['agent_status'] == 'moving' and info['mission_status'] == 'safety system':
                    print("__________________________________________")
                    print(" we lost the leader and waypoints ")
                    print("__________________________________________")
                    # action[0] *= 0
                    action[0] *= speed_coeff
                    if count_lost > 5:
                        camera_flag = _rotate_the_camera(camera_flag)

                if info['agent_status'] == 'too_close_from_leader_last_point':
                    print("__________________________________________")
                    print(" we lost the leader and waypoints ")
                    print("__________________________________________")
                    action[0] *= 0
                    if count_lost > 5:
                        camera_flag = _rotate_the_camera(camera_flag)

                # TODO : возврат ведущего в корридор

                # if list(obs[0:7]) == list(pus_obs):
                #     print("__________________________________________")
                #     print(" came out of the hallway ")
                #     print("__________________________________________")

                    # obs = env.reset()

                # if list(obs[0:7]) == list(pus_obs):
                # if action[0] <= 0.2 and count_lost > 16:
                #     print("__________________________________________")
                #     print(" we lost the leader and waypoints ")
                #     print("__________________________________________")

                    # action[0] *= 0
                    # camera_flag = _rotate_the_camera(camera_flag)

                # env.pub.set_camera_yaw(0)

                count_lost += 1
                if count_lost >= 1:
                    # Альтернативный вариант останавливать ведомого через 5 тактов даже если есть путевые точки
                    # if count_lost > 5:
                        # action[0] *= 0
                    # TODO : вернуть
                    print('')
                    pub.target_cancel_action()

            else:
                action[0] *= speed_coeff


            print(action)


            new_obs, reward, done, new_info = env.step(action)
            obs = new_obs
            info = new_info
            client.log_returns(eid, reward, info=info)
            rewards += reward

            if done:
                client.end_episode(eid, obs)
                break

    env.reset()
    env.close()
