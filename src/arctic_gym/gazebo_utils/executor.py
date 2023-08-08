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
        Перемещение ведущего и ведомого в позицию point_a
        Вычисление угла поворота в точку point_b

        :param point_a: стартовая точка маршрута в виде [x, y, z]
        :param point_b: конечная точка маршрута в виде [x, y, z]
        :param target_distance: расстояние между ведомым и ведущим в стартовой точке

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
        Следование за лидером в точку point_b

        :param point_b: конечная точка маршрута в виде [x, y, z]
        :return:
            Список [
                статус ведущего и ведомого, прогресс выполнения следования
                начальная точка маршрута
                конечная точка маршрута
                путь движения ведущего
                путь движения ведомого
            ]
        """
        start = self.env.sub.get_odom().pose
        point_a = [start.pose.position.x, start.pose.position.y, start.pose.position.z]

        # try:
        self.env.pub.move_target(*point_b[:2], self.phi)
        # except AttributeError:
        #     self.env.pub.move_target(*point_b[:2], 0)

        self.env.set_goal(point_b[:2])

        obs = self.env.reset(move=False)

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

        while not done:
            action = self.client.get_action(eid, obs)

            if info['leader_status'] == "moving" and count_lost >= 1:
                count_lost = 0
                # try:
                self.env.pub.move_target(*point_b[:2], self.phi)
                # except AttributeError:
                #     self.env.pub.move_target(*point_b[:2], 0)

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

            else:
                action *= 2.5

            new_obs, reward, done, new_info = self.env.step(action)
            obs = new_obs
            info = new_info
            self.client.log_returns(eid, reward, info=info)
            rewards += reward

            if done:
                self.client.end_episode(eid, obs)

        self.env.pub.set_camera_yaw(0)

        target_path = self.env.sub.get_target_path().poses
        follower_path = self.env.sub.get_robot_path().poses

        return [
            info,
            point_a,
            point_b,
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
