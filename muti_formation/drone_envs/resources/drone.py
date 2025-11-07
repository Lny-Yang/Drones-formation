import pybullet as p
import os
import numpy as np
import math
from ..config import multi_drone_env

class Drone:

    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), './drone.urdf')

        self.drone = p.loadURDF(
            fileName=f_name,
            basePosition=[0, 0, 0.1],
            physicsClientId=client
        )
        # 大幅提高角阻尼，最大限度减少物体自旋，降低线阻尼以便更好地控制高度
        try:
            p.changeDynamics(self.drone, -1,
                linearDamping=0.6,  # 🔥 从0.0增加到0.6 - 防止侧滑，确保移动方向=朝向
                angularDamping=15.0,  # 🔥 从3.0大幅增加到15.0 - 减少旋转惯性
                physicsClientId=self.client)
            
            # 对所有链接也应用相同的阻尼参数
            for i in range(p.getNumJoints(self.drone, physicsClientId=self.client)):
                p.changeDynamics(self.drone, i,
                    linearDamping=0.6,  # 🔥 从0.0增加到0.6
                    angularDamping=15.0,  # 🔥 从3.0大幅增加到15.0
                    physicsClientId=self.client)
        except Exception as e:
            print(f"Warning: Failed to set dynamics properties: {e}")

        # 自动查找 camera_link 的 index
        self.camera_link_index = None
        num_joints = p.getNumJoints(self.drone, physicsClientId=self.client)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.drone, i, physicsClientId=self.client)
            child_link_name = joint_info[12].decode('utf-8') if isinstance(joint_info[12], bytes) else joint_info[12]
            if child_link_name == 'camera_link':
                self.camera_link_index = i
                break

    def get_camera_pose(self):
        """
        获取摄像头 link 的位置和朝向（四元数）
        Returns: (pos, orn)
        """
        if self.camera_link_index is None:
            # fallback: 返回base_link
            return p.getBasePositionAndOrientation(self.drone, self.client)
        return p.getLinkState(self.drone, self.camera_link_index, physicsClientId=self.client)[:2]

    def get_ids(self):
        return self.drone, self.client

    def apply_action(self, action, apply_gravity_compensation=True):
        '''Apply body-frame thrust and turning torque for planar movement with camera-based obstacle avoidance.
        
        Args:
            action: [thrust, torque] - thrust along current heading, torque for turning
        '''
        thrust, torque = action[0], action[1]
        
        # Clip using bounds from config
        thrust = float(np.clip(thrust, 
                               multi_drone_env['thrust_lower_bound'], 
                               multi_drone_env['thrust_upper_bound']))
        torque = float(np.clip(torque, 
                               multi_drone_env['torque_lower_bound'], 
                               multi_drone_env['torque_upper_bound']))

        # 获取当前姿态
        _, orientation = p.getBasePositionAndOrientation(self.drone, self.client)
        euler_angles = p.getEulerFromQuaternion(orientation)
        
        # 只使用yaw角度（水平转向）来确定前进方向
        # 忽略pitch和roll，因为在planar模式下它们被强制为0
        yaw = euler_angles[2]
        
        # 计算水平前进方向（基于yaw角度）
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        
        # 沿着水平朝向施加推力（摄像头方向）
        force_x = thrust * forward_x if abs(thrust) > 1e-6 else 0
        force_y = thrust * forward_y if abs(thrust) > 1e-6 else 0
        
        # 【确保】planar模式下不产生z方向力
        force_z = 0
        
        if abs(thrust) > 1e-10:
            # 🔧 关键修复：获取drone质心的世界坐标
            # 这样力施加在质心上，不会产生pitch/roll力矩
            com_pos, _ = p.getBasePositionAndOrientation(self.drone, self.client)
            
            p.applyExternalForce(
                objectUniqueId=self.drone,
                linkIndex=-1,
                forceObj=[force_x, force_y, force_z],
                posObj=com_pos,  # 🔥 使用质心的世界坐标
                flags=p.WORLD_FRAME,
                physicsClientId=self.client
            )
        
        # Apply turning torque around Z-axis (allows camera scanning)
        if abs(torque) > 1e-10:
            p.applyExternalTorque(
                objectUniqueId=self.drone,
                linkIndex=-1,
                torqueObj=[0, 0, torque],
                flags=p.WORLD_FRAME,
                physicsClientId=self.client
            )
        
        # 【移除】重力补偿逻辑，统一由环境处理


    def get_observation(self):
        pos, ang = p.getBasePositionAndOrientation(self.drone, self.client)
        ang = p.getEulerFromQuaternion(ang)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.drone, self.client)
        observation = (pos + linear_velocity)
        return observation

    def get_forward_speed(self):
        """获取无人机沿当前朝向的前进速度(用于固定翼行为监控)
        
        Returns:
            float: 前进速度 (m/s), 正值为前进, 负值为后退
        """
        # 获取当前姿态和速度
        _, orientation = p.getBasePositionAndOrientation(self.drone, self.client)
        linear_velocity, _ = p.getBaseVelocity(self.drone, self.client)
        
        # 获取yaw角度
        euler_angles = p.getEulerFromQuaternion(orientation)
        yaw = euler_angles[2]
        
        # 计算前进方向单位向量
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        
        # 计算速度在前进方向的投影(点积)
        forward_speed = linear_velocity[0] * forward_x + linear_velocity[1] * forward_y
        
        return forward_speed
    
    def get_horizontal_speed(self):
        """获取无人机在水平面的总速度
        
        Returns:
            float: 水平速度 (m/s)
        """
        linear_velocity, _ = p.getBaseVelocity(self.drone, self.client)
        # 计算水平速度(忽略z方向)
        horizontal_speed = math.sqrt(linear_velocity[0]**2 + linear_velocity[1]**2)
        return horizontal_speed
