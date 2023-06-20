import pdal
import os
import tf
import time
import rospy
import numpy as np

from pyhocon import ConfigFactory
from ray.rllib.env.policy_client import PolicyClient
from flask import Flask, request

from src.arctic_gym import arctic_env_maker


PATH_TO_CONFIG = os.path.join(os.path.expanduser('~'), 'arctic_build', 'continuous-grid-arctic', 'config', 'config.conf')
config = ConfigFactory.parse_file(PATH_TO_CONFIG)


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


def _get_default_status():
    default_status_to = env.sub.get_move_to_status()
    try:
        code_to, text_to = default_status_to.status_list[-1].status, default_status_to.status_list[-1].text
    except IndexError as e:
        code_to = 000
    return code_to


def _go_to_the_leader(leader_pose):
    env.pub.target_cancel_action()

    data = env.sub.get_odom_target()

    quaternion = data.pose.pose.orientation
    orientation = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    _, _, yaw = tf.transformations.euler_from_quaternion(orientation)
    arctic_goal = [leader_pose[0] - 9 * np.cos(yaw), leader_pose[1] - 9 * np.sin(yaw)]
    env.pub.move_default(arctic_goal[0], arctic_goal[1], phi=np.rad2deg(yaw))

    code_to = _get_default_status()

    return code_to


if __name__ == '__main__':
    rospy.init_node("rl_client", anonymous=True)

    client = PolicyClient(config.rl_agent.get_weights, inference_mode='remote')
    env_config = config.rl_agent.env_config
    env = arctic_env_maker(env_config)

    camera_flag = True
    count_lost = 0

    obs = env.reset(move=True)

    done = False
    rewards = 0
    lead_loss = False
    pub_obs = np.ones(7, dtype=np.float32)
    info = {
        "mission_status": "in_progress",
        "agent_status": "None",
        "leader_status": "moving"
    }

    eid = client.start_episode(training_enabled=True)

    lead_pose = env.leader_position
    arc_pose = env.follower_position

    lead_info = env.ssd_camera_objects
    lead_info = next((x for x in lead_info if x["name"] == "car"), None)

    # if lead_info == None:
    # follower_status = _go_to_the_leader(lead_pose)
    # while True:
    #     code_to = _get_default_status()
    #     follower_status = code_to
    #     if follower_status == 3:
    env.pub.move_target(50.0, -40.0, phi=90)
    #         break

    # time.sleep(1)
    # env.pub.text_to_voice('Начинаю следовать за машиной')

    obs = env.reset(move=False)

    while not done:
        action = client.get_action(eid, obs)
        # time.sleep(10)

        if info['leader_status'] == "moving" and count_lost >= 1:
            count_lost = 0
            env.pub.move_target(50.0, -40.0, phi=90)

        if info['agent_status'] == 'too_far_from_leader_info':
            count_lost += 1
            env.pub.target_cancel_action()

        if info["agent_status"] == "too_close_to_leader" and info["leader_status"] == 'moving':
            action[0] *= 0

        if info['leader_status'] == 'None':
            if list(obs[0:7]) == list(pub_obs):
                action[0] *= 0
                camera_flag = _rotate_the_camera(camera_flag)

            count_lost += 1
            if count_lost >= 2:
                env.pub.target_cancel_action()

        else:
            action[0] *= 1.5

        new_obs, reward, done, new_info = env.step(action)
        obs = new_obs
        info = new_info
        client.log_returns(eid, reward, info=info)
        rewards += reward

        if done:
            client.end_episode(eid, obs)
            # if info['mission_status'] == 'fail':
            #     if info['agent_status'] == 'low_reward':
            #         env.pub.text_to_voice('Невозможно вернутся на маршрут следования')
            #     elif info['agent_status'] == 'too_far_from_leader':
            #         env.pub.text_to_voice('Дистанция до машины слишком велика')
            # elif info['mission_status'] == 'safety system end':
            #     env.pub.text_to_voice('Машина не обнаружена')
            # elif info['mission_status'] == 'success':
            #     env.pub.text_to_voice('Следование завершено успешно')

    env.pub.set_camera_yaw(0)
