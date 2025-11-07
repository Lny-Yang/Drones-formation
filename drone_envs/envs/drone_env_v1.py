import gym
import numpy as np
import math
import pybullet as p
from ..resources.drone import Drone
from ..resources.plane import Plane
from ..resources.goal import Goal
import time
from collections import deque
from ..config import drone_env_v1 as config
from ..config import observation_space_v1 as observation_space
import random
import matplotlib.pyplot as plt
import pybullet_data
from agent.PPOagent import PPO

class DroneNavigationV1(gym.Env):
    ## 定义环境支持的渲染模式
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        无人机动作空间说明:
            - 无人机 z 轴方向期望推力
            - 无人机绕 x 轴期望扭矩
            - 无人机绕 y 轴期望扭矩
            - 无人机绕 z 轴期望扭矩
        """
        # 定义动作空间，指定动作的上下界
        self.action_space = gym.spaces.box.Box(
            low=np.array([config['thrust_x_lower_bound'], 
                          config['thrust_y_lower_bound'], 
                          config['thrust_z_lower_bound']], dtype=np.float32),
            high=np.array(
                [config['thrust_x_upper_bound'], 
                 config['thrust_y_upper_bound'], 
                 config['thrust_z_upper_bound']], dtype=np.float32)
        )
        # 定义观测空间，指定观测值的上下界
        self.observation_space = gym.spaces.box.Box(
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
        # 初始化随机数生成器
        self.np_random, _ = gym.utils.seeding.np_random()

        # 连接到 PyBullet 物理模拟客户端
        self.client = p.connect(config["display"])        
        # 为强化学习算法缩短回合时长，设置模拟时间步长
        p.setTimeStep(1/30, self.client)
        # 初始化障碍物位置列表
        self.obstacles_pos_list = []
        # 初始化无人机对象
        self.drone = None
        # 初始化障碍物列表
        self.obstacle_list = None
        # 初始化目标位置
        self.goal = None
        # 标记回合是否结束
        self.done = False
        # 记录上一次无人机到目标的距离
        self.prev_dist_to_goal = None
        # 用于存储渲染图像的对象
        self.rendered_img = None
        # 重置 PyBullet 物理模拟环境
        p.resetSimulation(self.client)
        # 设置环境中的障碍物
        self.setup_obstacles()
        # 初始化目标对象的 ID
        self.goal_id = None
        # 设置相机像素值
        self.camera_pixel = config["camera_pixel"]
        # 标记是否到达目标
        self.reach_target = False
        
    def step(self, action):
        """
        执行一个动作步骤，更新环境状态并返回观测、奖励等信息。

        参数:
        action (list): 无人机的动作

        返回:
        tuple: 包含观测、奖励、是否结束标志和额外信息的元组
        """
        # 将动作应用到无人机上
        self.drone.apply_action(action)
        # 推进 PyBullet 物理模拟一步
        p.stepSimulation()
        # 获取无人机的当前观测信息
        drone_ob = self.drone.get_observation()
        # 计算当前步骤的奖励
        reward = self.calculate_reward(drone_ob)
        # 计算无人机当前位置到目标的距离
        dist_to_goal = self.calculate_distance_from_goal(drone_ob)
        # 更新上一次无人机到目标的距离
        self.prev_dist_to_goal = dist_to_goal

        # 获取无人机相机图像
        image = self.get_drone_camera_image()
        # 将无人机观测信息和目标位置组合成元数据
        metadata = np.array(drone_ob + self.goal, dtype=np.float32)
        # 将图像数据和元数据拼接成观测信息
        ob = np.concatenate((image.flatten(), metadata))
        # 包含额外信息的字典，记录是否到达目标
        info = {"reach_target": self.reach_target}

        return ob, reward, self.done, info

    def seed(self, seed=None):
        """
        设置随机数种子，保证实验可复现。

        参数:
        seed (int, optional): 随机数种子

        返回:
        list: 包含随机数种子的列表
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        重置环境到初始状态。

        返回:
        np.ndarray: 初始观测信息
        """
        # 打印重置环境的提示信息
        print("reset env")
        # 标记回合未结束
        self.done = False
        # 标记未到达目标
        self.reach_target = False
        # 重新加载平面环境
        Plane(self.client)
        # 如果无人机对象已存在，移除该对象
        if self.drone is not None:
            p.removeBody(self.drone.drone, self.client)
        # 重新创建无人机对象
        self.drone = Drone(self.client)
        # 随机重置目标位置
        self.reset_goal_position()
        # 获取无人机的初始观测信息
        drone_ob = self.drone.get_observation()
        # 调用渲染方法并重置图像队列
        self.render()
        # 获取无人机相机图像
        image = self.get_drone_camera_image()
        # 计算无人机初始位置到目标的距离
        self.prev_dist_to_goal = self.calculate_distance_from_goal(drone_ob)
        # 将无人机观测信息和目标位置组合成元数据
        metadata = np.array(drone_ob + self.goal, dtype=np.float32)
             
        return np.concatenate((image.flatten(), metadata))

    def render(self, mode = 'human'):
        """
        渲染环境，获取无人机视角的图像。

        参数:
        mode (str, optional): 渲染模式，默认为 'human'
        """
        # 获取无人机的 ID 和客户端 ID
        drone_id, client_id = self.drone.get_ids()
        # 计算投影矩阵，用于将 3D 场景投影到 2D 平面
        proj_matrix = p.computeProjectionMatrixFOV(fov=100, aspect=1,
                                                   nearVal=0.01, farVal=100)
        # 获取无人机的位置和方向（四元数表示）
        pos, ori = p.getBasePositionAndOrientation(drone_id, client_id)
        # 获取无人机的线速度
        linear_vel, _ = p.getBaseVelocity(drone_id)
        # 计算视图矩阵，确定相机的视角
        view_matrix = p.computeViewMatrix(
                                      (pos[0] + 0.1, pos[1],pos[2]+0.05), 
                                      (pos[0] + linear_vel[0], pos[1] + linear_vel[1], 
                                       pos[2] + linear_vel[2]), 
                                      [0, 0, 1]
                                      )
        # 计算偏航角
        yaw = math.atan2(linear_vel[0], linear_vel[2])
        # 计算俯仰角
        pitch = math.atan2(linear_vel[1], linear_vel[2])
        # 重置调试可视化相机的参数
        p.resetDebugVisualizerCamera(cameraDistance = 0.5, cameraYaw=yaw, cameraPitch=pitch,cameraTargetPosition=pos)
        
        # 获取相机图像，返回值的第四个元素是深度图像数据
        frame = p.getCameraImage(self.camera_pixel, 
                                 self.camera_pixel, 
                                 view_matrix, proj_matrix)[3]
        # 将图像数据重新调整为指定像素大小的二维数组
        frame = np.reshape(frame, (self.camera_pixel, self.camera_pixel, 1))
        
        # 设置当前帧图像
        self.frame = frame

    def close(self):
        """
        断开与 PyBullet 物理模拟客户端的连接。
        """
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
        """
        随机重置目标位置，并创建目标的可视化元素。

        返回:
        tuple: 目标的位置坐标
        """
        # 如果目标对象已存在，移除该对象
        if self.goal_id is not None:
            p.removeBody(self.goal_id, self.client)
        # 随机生成目标的 x 坐标，有 50% 的概率在正半轴，50% 的概率在负半轴
        x = (self.np_random.uniform(8, 20) if self.np_random.integers(2) else
             self.np_random.uniform(-20, -8))
        # 随机生成目标的 y 坐标，有 50% 的概率在正半轴，50% 的概率在负半轴
        y = (self.np_random.uniform(8, 20) if self.np_random.integers(2) else
             self.np_random.uniform(-20, -8))
        # 随机生成目标的 z 坐标
        z = self.np_random.uniform(5, 12)
        # 设置目标位置
        self.goal = (x, y, z)

        # 创建目标的可视化元素并获取其 ID
        self.goal_id = Goal(self.client, self.goal).id
        return self.goal

    def calculate_reward(self, observation):
        """
        计算当前步骤的奖励。

        参数:
        observation (list): 无人机的观测信息

        返回:
        float: 奖励值
        """
        # 计算无人机当前位置到目标的距离
        distance = self.calculate_distance_from_goal(observation)
        # 计算与上一时刻相比，无人机到目标的距离改进量
        distance_improvement = self.prev_dist_to_goal - distance
        # 奖励值初始化为距离改进量
        reward = distance_improvement
        # 打印奖励值（当前为注释状态）
        # print(reward)
        # 判断无人机是否飞出边界，如果是则给予惩罚并结束回合
        if (observation[0] >= 28 or observation[0] <= -28 or
                observation[1] >= 28 or observation[1] <= -28 or
                observation[2] <= 0.01 or observation[2] >= 20):
            reward -= 0.5
            self.done = True

        # 判断无人机是否到达目标，如果是则给予奖励并结束回合
        if distance < 2:
            self.done = True
            self.reach_target = True
            # 打印到达目标的提示信息和时间戳
            print("reach the goal!  timestamp-" + str(time.time()))
            reward += 5
        
        # 检查是否发生碰撞，如果发生则给予惩罚
        if self.check_collisions(self.drone.drone):
            reward -= 0.5
        return reward
    
    def setup_obstacles(self, obstacle_num = 30):
        """
        在环境中设置障碍物。

        参数:
        obstacle_num (int, optional): 障碍物的数量，默认为 30

        返回:
        list: 障碍物对象的列表
        """
        # 设置 PyBullet 数据的搜索路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        obstacle_list = []
        # 设置随机数种子
        random.seed(23)
        for _ in range(obstacle_num):
            # 随机生成障碍物的位置
            obstacle_pos = [random.randint(-12,12), random.randint(-12,12), random.randint(2,9)]
            # 打印障碍物位置（当前为注释状态）
            # print(obstacle_pos)
            #结合p.setAdditionalSearchPath(pybullet_data.getDataPath()) 这一行，cube.urdf 实际上是 PyBullet 数据目录下的一个文件，代表一个立方体模型。
            cube = p.loadURDF("cube.urdf", basePosition=obstacle_pos)
            # 将障碍物对象添加到列表中
            obstacle_list.append(cube)
            # 将障碍物位置添加到位置列表中
            self.obstacles_pos_list.append(obstacle_pos)
        # 更新障碍物列表
        self.obstacle_list = obstacle_list
        return self.obstacle_list

    def check_collisions(self, object):
        """
        检查指定对象是否与任何障碍物发生碰撞。

        参数:
        object: 需要检查碰撞的对象

        返回:
        bool: 如果发生碰撞返回 True，否则返回 False
        """
        # 如果存在接触点，则认为发生碰撞
        return True if p.getContactPoints(object) else False
    
    def get_drone_camera_image(self):
        """
        获取无人机的相机图像。

        返回:
        np.ndarray: 相机图像数据
        """
        return self.frame