import pybullet as p
import os
import numpy as np

class Drone:
    def __init__(self, client):
        #client 参数代表 PyBullet 物理客户端的 ID，用于指定模拟环境。
        #self.client = client：将传入的客户端 ID 存储为实例属性。
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), './drone.urdf')

        #p.loadURDF：加载 URDF 文件，将无人机模型添加到 PyBullet 模拟环境中。basePosition=[0, 0, 0.1] 表示无人机初始位置在 (0, 0, 0.1) 处。
        self.drone = p.loadURDF(
            fileName=f_name,
            basePosition=[0, 0, 0.1],
            physicsClientId=client
        )

    #返回无人机在 PyBullet 中的唯一标识符和客户端 ID，方便外部获取这两个关键信息。
    def get_ids(self):
        return self.drone, self.client

    def apply_action(self, action):
        """施加带重力补偿的外力：假设模型质量 m，提供 m*g 的基线推力。"""
        thrust_x, thrust_y, thrust_z = action
        # 放宽裁剪范围，匹配新配置
        thrust_x = float(np.clip(thrust_x, -1.0, 1.0))
        thrust_y = float(np.clip(thrust_y, -1.0, 1.0))
        thrust_z = float(np.clip(thrust_z, -1.0, 1.5))

        # 估计质量：读取 base link 质量（link -1）
        dyn = p.getDynamicsInfo(self.drone, -1)
        mass = dyn[0] if dyn else 1.0
        g = 9.8
        # 分配基线推力到 z 方向（世界坐标），此处简单采用世界坐标力
        baseline_z = mass * g
        # 将智能体动作的 z 分量视为增量比例（-1~1.5 映射到 [-baseline_z, 1.5*baseline_z] 的增量），可简化为系数
        # 为避免过大，使用缩放系数 scale
        scale = baseline_z
        force_world = [thrust_x * scale * 0.2, thrust_y * scale * 0.2, baseline_z + thrust_z * scale * 0.3]
        # 直接在 base 施加世界坐标力
        p.applyExternalForce(
            self.drone,
            -1,
            forceObj=force_world,
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
        )


    def get_observation(self):
        # Get the position and orientation of the car in the simulation
        #p.getBasePositionAndOrientation 获取无人机的位置和四元数表示的姿态。
        pos, ang = p.getBasePositionAndOrientation(self.drone, self.client)

        #p.getEulerFromQuaternion 将四元数转换为欧拉角。
        ang = p.getEulerFromQuaternion(ang)

        # p.getBaseVelocity 获取无人机的线速度和角速度。
        linear_velocity, angular_velocity = p.getBaseVelocity(self.drone, self.client)
        # print("ob: ", ang, "vel: ", p.getBaseVelocity(self.drone, self.client))
        
        #observation = (pos + linear_velocity) 将位置和线速度组合成一个元组作为观测值返回，不过这里注释提到获取姿态但实际未使用欧拉角，且未使用角速度。
        observation = (pos + linear_velocity)
        return observation









