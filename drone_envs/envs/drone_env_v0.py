import gym
import numpy as np
import math
import pybullet as p
from ..resources.drone import Drone
from ..resources.plane import Plane
from ..resources.goal import Goal
import time
from ..config import drone_env_v0 as config
from ..config import observation_space_v0 as observation_space
import matplotlib.pyplot as plt


class DroneNavigationV0(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        drone action space:
            - The desired thrust along the drone's x-axis
            - The desired thrust along the drone's y-axis
            - The desired thrust along the drone's z-axis
        """
        self.action_space = gym.spaces.box.Box(
            low=np.array([-0.1, -0.1, -0.1], dtype=np.float32),
            high=np.array([0.1, 0.1, 0.1], dtype=np.float32)
        )
        # 初始化观测空间，定义无人机状态和目标位置的上下界
        self.observation_space = gym.spaces.box.Box(
            # 观测空间的下界，包含无人机位置、速度和目标位置的下限
            low=np.array([observation_space["drone_lower_bound_x"], 
                          observation_space["drone_lower_bound_y"], 
                          observation_space["drone_lower_bound_z"], 
                          observation_space["drone_velocity_lower_bound_x"],
                          observation_space["drone_velocity_lower_bound_y"],
                          observation_space["drone_velocity_lower_bound_z"],
                          observation_space["goal_lower_bound_x"],
                          observation_space["goal_lower_bound_y"],
                          observation_space["goal_lower_bound_z"],
                          ], dtype=np.float32),
            # 观测空间的上界，包含无人机位置、速度和目标位置的上限
            high=np.array([observation_space["drone_upper_bound_x"], 
                          observation_space["drone_upper_bound_y"], 
                          observation_space["drone_upper_bound_z"], 
                          observation_space["drone_velocity_upper_bound_x"],
                          observation_space["drone_velocity_upper_bound_y"],
                          observation_space["drone_velocity_upper_bound_z"],
                          observation_space["goal_upper_bound_x"],
                          observation_space["goal_upper_bound_y"],
                          observation_space["goal_upper_bound_z"],
                          ], dtype=np.float32))
        # 初始化随机数生成器，用于后续的随机操作
        self.np_random, _ = gym.utils.seeding.np_random()

        # 连接到 PyBullet 物理模拟客户端，显示模式根据配置文件决定
        self.client = p.connect(config["display"])
        # 为强化学习算法缩短回合时长，设置模拟的时间步长为 1/30 秒
        p.setTimeStep(1/30, self.client)

        # 初始化无人机对象，初始值为 None
        self.drone = None
        # 初始化目标位置，初始值为 None
        self.goal = None
        # 标记回合是否结束，初始值为 False
        self.done = False
        # 记录上一次无人机到目标的距离，初始值为 None
        self.prev_dist_to_goal = None
        # 用于存储渲染图像的对象，初始值为 None
        self.rendered_img = None
        # 用于存储渲染时的旋转矩阵，初始值为 None
        self.render_rot_matrix = None
        # 标记是否到达目标，初始值为 False
        self.reach_target = False
        # 调用 reset 方法重置环境
        self.reset()

    def step(self, action):
        # 将动作应用到无人机上
        self.drone.apply_action(action)
        # 推进 PyBullet 物理模拟一步
        p.stepSimulation()
        # 获取无人机的当前观测信息
        drone_ob = self.drone.get_observation()
        # 计算当前步骤的奖励，奖励基于无人机到目标距离的变化
        reward = self.calculate_reward(drone_ob)
        # 打印奖励值（当前为注释状态）
        # print(reward)
        # 计算无人机当前位置到目标的距离
        dist_to_goal = self.calculate_distance_from_goal(drone_ob)
        # 更新上一次无人机到目标的距离
        self.prev_dist_to_goal = dist_to_goal

        # 将无人机观测信息和目标位置组合成一个观测数组
        ob = np.array(drone_ob + self.goal, dtype=np.float32)
        # 包含额外信息的字典，记录是否到达目标
        info = {"reach_target": self.reach_target}
        
        # 返回观测、奖励、回合是否结束标志和额外信息
        return ob, reward, self.done, info

    def seed(self, seed=None):
        # 设置随机数种子，确保实验的可重复性
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        # 打印重置环境的提示信息
        print("reset")
        # 重置 PyBullet 物理模拟环境
        p.resetSimulation(self.client)
        # 标记回合未结束
        self.done = False
        # 标记未到达目标
        self.reach_target = False
        
        # 重新加载平面环境
        Plane(self.client)
        # 重新创建无人机对象
        self.drone = Drone(self.client)
        # 随机重置目标位置
        self.reset_goal_position()
        # 获取无人机的初始观测信息
        drone_ob = self.drone.get_observation()

        # 计算无人机初始位置到目标的距离
        self.prev_dist_to_goal = self.calculate_distance_from_goal(drone_ob)
        # 返回包含无人机观测信息和目标位置的初始观测数组
        return np.array(drone_ob + self.goal, dtype=np.float32)

    def render(self, mode='human'):
        # 如果渲染图像对象还未初始化，则创建一个初始的图像对象
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # 获取无人机的 ID 和客户端 ID
        drone_id, client_id = self.drone.get_ids()
        # 计算投影矩阵，用于将 3D 场景投影到 2D 平面
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        # 获取无人机的位置和方向（四元数表示）
        pos, ori = p.getBasePositionAndOrientation(drone_id, client_id)

        # 将四元数转换为旋转矩阵
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        # 计算相机的视线方向向量
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        # 计算相机的向上方向向量
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        # 计算视图矩阵，确定相机的视角
        view_matrix = p.computeViewMatrix((pos[0], pos[1],pos[2]+0.05), pos + camera_vec, up_vec)

        # 获取相机图像，返回值的第三个元素是 RGB 图像数据
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        # 将图像数据重新调整为 100x100x4 的形状
        frame = np.reshape(frame, (100, 100, 4))
        # 更新渲染图像对象的数据
        self.rendered_img.set_data(frame)
        # 重绘图像
        plt.draw()

    def close(self):
        # 断开与 PyBullet 物理模拟客户端的连接
        p.disconnect(self.client)

    def calculate_distance_from_goal(self, observation):
        """
        计算无人机到目标的欧几里得距离。

        参数:
        observation (list): 无人机的观测信息，包含位置信息

        返回:
        float: 无人机到目标的欧几里得距离
        """
        # 提取无人机的位置信息
        drone_pos = [observation[0], observation[1], observation[2]]
        
        return math.sqrt(
            (drone_pos[0] - self.goal[0]) ** 2 +
            (drone_pos[1] - self.goal[1]) ** 2 +
            (drone_pos[2] - self.goal[2]) ** 2
        )

    def reset_goal_position(self):
        # 随机生成目标的 x 坐标，有 50% 的概率在正半轴，50% 的概率在负半轴
        x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        # 随机生成目标的 y 坐标，有 50% 的概率在正半轴，50% 的概率在负半轴
        y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        # 随机生成目标的 z 坐标
        z = self.np_random.uniform(5, 9)
        # 设置目标位置
        self.goal = (x, y, z)

        # 创建目标的可视化元素
        Goal(self.client, self.goal)
        # 返回目标位置
        return self.goal

    def calculate_reward(self, observation):
        # 打印无人机的观测信息（当前为注释状态）
        # print(observation)
        # 计算无人机当前位置到目标的距离
        distance = self.calculate_distance_from_goal(observation)
        # 计算与上一时刻相比，无人机到目标的距离改进量
        distance_improvement = self.prev_dist_to_goal - distance
        # 奖励值初始化为距离改进量
        reward = distance_improvement

        # 判断无人机是否飞出边界，如果是则给予惩罚并结束回合
        if (observation[0] >= 12 or observation[0] <= -12 or
                observation[1] >= 12 or observation[1] <= -12 or
                observation[2] <= 0 or observation[2] >= 12):
            reward -= 1
            self.done = True

        # 判断无人机是否到达目标，如果是则给予奖励并结束回合
        if distance < 2:
            self.done = True
            self.reach_target = True
            # 打印到达目标的提示信息和时间戳
            print("reach the goal!  timestamp-" + str(time.time()))
            reward += 50
        
        return reward
    