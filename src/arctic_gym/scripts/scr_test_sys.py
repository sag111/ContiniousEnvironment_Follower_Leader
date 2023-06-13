#!/usr/bin/env python

import rospy
import numpy as np
from src.arctic_gym import arctic_env_maker
from ray.rllib.env.policy_client import PolicyClient


def _get_default_status():
    default_status_to = env.sub.get_move_to_status()
    try:
        code_to, text_to = default_status_to.status_list[-1].status, default_status_to.status_list[-1].text
        # print("STATUS TO: ", code_to, text_to)

    except IndexError as e:
        print(f"Проблема получения статус ведомго: {e}")
        code_to = 000
    print("")
    # print(code_to)
    return code_to

def _go_to_the_leader(leader_pose):

    env.pub.target_cancel_action()

    arctic_goal = [leader_pose[0]-12, leader_pose[1]]
    env.pub.move_default(arctic_goal[0], arctic_goal[1])

    code_to = _get_default_status()

    return code_to

def _rotate_the_camera(camera_flag):

    camera_yaw_state_info = env.sub.get_camera_yaw_state()
    rot_cam = camera_yaw_state_info.process_value

    test_camera = abs(rot_cam - 3.59758)
    test_camera_1 = abs(rot_cam + 3.59758)

    if test_camera < 0.7:
        camera_flag = False

    if test_camera_1 < 0.5:
        camera_flag = True

    if camera_flag:
        rot_cam = rot_cam + 0.456
    else:
        rot_cam = rot_cam - 0.456

    env.pub.set_camera_yaw(rot_cam)

    return camera_flag


if __name__ == '__main__':
    rospy.init_node('arctic_gym_rl', anonymous=True, log_level=rospy.INFO)

    env_config = {"name": "ArcticRobot-v1",
                  "discrete_action": False,
                  "time_for_action": 0.5}
    env = arctic_env_maker(env_config)
    # sub = Subscribers()
    # pub = Publishers()

    # TODO: ip
    client = PolicyClient("http://10.8.0.10:9900", inference_mode='remote')
    # client = PolicyClient("http://localhost:9900", inference_mode='remote')

    # goal = [30.0, -50.0]
    goal = env.goal

    # camera_yaw = 0
    count_lost = 0

    rot_cam = 0




    while True:


        obs = env.reset()
        eid = client.start_episode(training_enabled=True)
        rewards = 0
        lead_loss = False
        pus_obs = np.zeros(7, dtype=np.float32)
        info = {
            "mission_status": "in_progress",
            "agent_status": "None",
            "leader_status": "moving"
        }

        # lead_pose = env.leader_position
        # arctic_goal = [lead_pose[0]-10, lead_pose[1]]

        camera_flag = True

        lead_pose = env.leader_position
        print("LEADELADLEAD ; ", lead_pose)
        arc_pose = env.follower_position

        lead_info = env.ssd_camera_objects
        lead_info = next((x for x in lead_info if x["name"] == "car"), None)
        print("LEAD INFO", lead_info)

        # default_status_to = env.sub.get_move_to_status()
        # try:
        #     code_to, text_to = default_status_to.status_list[-1].status, default_status_to.status_list[-1].text
        #     print("STATUS TO: ", code_to, text_to)
        #
        # except IndexError as e:
        #     print(f"Проблема получения статус ведомго: {e}")
        # print("")
        # print(code_to)

        if lead_info == None or lead_info['length'] > 10:
            # Перемещение к лидеру
            follower_status = _go_to_the_leader(lead_pose)
            print("STATUS ", follower_status)
            while True:
                code_to = _get_default_status()

                follower_status = code_to
                print(follower_status)
                print("WAIT")
                if follower_status == 3:
                    env.pub.move_target(goal[0], goal[1])
                    break




        for s in range(1000):

            action = client.get_action(eid, obs)
            # TODO : вернуть
            # action[0] *= 0

            if info['leader_status'] == "moving" and count_lost >= 1:
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

            if info['agent_status'] == 'too_far_from_leader_info':
                print("__________________________________________")
                print(" too far to the leader info ")
                print("__________________________________________")
                count_lost += 1
                env.pub.target_cancel_action()



            if info['leader_status'] == 'None':

                if list(obs[0:7]) == list(pus_obs):
                    print("__________________________________________")
                    print(" we lost the leader and waypoints ")
                    print("__________________________________________")
                    action[0] *= 0
                    # Начинаем поиск лидера
                    camera_flag = _rotate_the_camera(camera_flag)

                print("__________________________________________")
                print("we lost the leader")
                print("__________________________________________")
                # env.pub.set_camera_yaw(0)

                count_lost += 1
                if count_lost >= 3:
                    # Альтернативный вариант останавливать ведомого через 5 тактов даже если есть путевые точки
                    # if count_lost > 5:
                        # action[0] *= 0
                    env.pub.target_cancel_action()

            else:
                action[0] *= 1.7



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
