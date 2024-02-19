import json
import time

import numpy as np
import rospy

from ray.rllib.env.policy_client import PolicyClient

from src.arctic_gym import arctic_env_maker


class Executor:

    def __init__(self, config):
        rospy.init_node("rl_client", anonymous=True)

        env_config = config.rl_agent.env_config
        self.env = arctic_env_maker(env_config)

        self.client = PolicyClient(config.rl_agent.get_weights, inference_mode='remote')

    def setup_position(self, point_a: list, point_b: list, target_distance: float = 12):
        """
        Moves the leader and the agent to position point_a
        Calculates of the angle of rotation to point point_b

        :param point_a: starting point of the route in the form [x, y, z]
        :param point_b: route end point in the form [x, y, z]
        :param target_distance: distance between the agent and the leader at the starting point
        """
        self.env.pub.target_cancel_action()
        self.env.pub.set_camera_yaw(0)

        point_a = np.array(point_a)
        point_b = np.array(point_b)

        self.phi = np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0])
        orientation = [0, 0, np.sin(self.phi/2), np.cos(self.phi/2)]

        point_target = point_a + [target_distance*np.cos(self.phi), target_distance*np.sin(self.phi), 0.0]

        self.env.pub.teleport(model="arctic_model", point=point_a, quaternion=orientation)
        self.env.pub.teleport(model="target_robot", point=point_target, quaternion=orientation)

    def follow(self, point_b: list) -> list:
        """
        Following the leader to point_b

        :param point_b: route end point in the form [x, y, z]
        :return:
            List [
                the leader and the agent status, follow progress
                spent time
                route start point
                route end point
                leader's path
                agent's path
            ]
        """

        start_time = time.time()

        start = self.env.sub.get_odom().pose
        point_a = [start.pose.position.x, start.pose.position.y, start.pose.position.z]

        self.env.pub.move_target(*point_b[:2], self.phi)

        self.env.set_goal(point_b[:2])

        obs = self.env.reset()

        camera_flag = True
        done = False
        count_lost = 0
        rewards = 0
        pub_obs = np.ones(7, dtype=np.float32)
        info = {
            "mission_status": "in_progress",
            "agent_status": "None",
            "leader_status": "moving"
        }

        eid = self.client.start_episode(training_enabled=True)

        dynamic_states = []
        while not done:
            action = self.client.get_action(eid, obs)

            if info['leader_status'] == "moving" and count_lost >= 1:
                count_lost = 0
                self.env.pub.move_target(*point_b[:2], self.phi)

            if info['agent_status'] == 'too_far_from_leader_info':
                count_lost += 1
                self.env.pub.target_cancel_action()

            if info["agent_status"] == "too_close_to_leader":
                action[0] *= 0

            if info['leader_status'] == 'None':
                if list(obs[0:7]) == list(pub_obs):
                    action[0] *= 0
                    camera_flag = self._rotate_the_camera(camera_flag)

                count_lost += 1
                if count_lost >= 2:
                    self.env.pub.target_cancel_action()

            # action[0] = 2.00 * action[0]
            # action[1] = 1.50 * action[1]

            new_obs, reward, done, new_info = self.env.step(action)
            obs = new_obs
            info = new_info
            self.client.log_returns(eid, reward, info=info)
            rewards += reward

            # подсчет количества попаданий динамических объектов в область потенциального столкновения
            # в зависимости от длины луча сенсора 2
            sensor_2_length = 15
            dynamic_xys, robot_xy = self.env.get_actor_states()
            states = np.linalg.norm(dynamic_xys - robot_xy, axis=1) <= sensor_2_length
            dynamic_states.append(states.tolist())

            if done:
                self.client.end_episode(eid, obs)

        dynamic_states = json.dumps(dynamic_states)

        self.env.pub.set_camera_yaw(0)

        target_path = self.env.sub.get_target_path().poses
        follower_path = self.env.sub.get_robot_path().poses

        finish_time = time.time() - start_time

        return [
            info,
            finish_time,
            point_a,
            point_b,
            dynamic_states,
            target_path,
            follower_path
        ]

    def _rotate_the_camera(self, camera_flag):
        camera_yaw_state_info = self.env.sub.get_camera_yaw_state()
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

        self.env.pub.set_camera_yaw(rot_cam)

        return camera_flag
