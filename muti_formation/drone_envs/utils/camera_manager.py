"""
相机设置和图像获取模块
"""
import numpy as np
import pybullet as p
from typing import Dict, Any, Tuple, Optional


class CameraManager:
    
    def __init__(self, client: int, config: Dict[str, Any]):
        """
        初始化相机管理器
        
        Args:
            client: PyBullet客户端ID
            config: 相机配置参数
        """
        self.client = client
        self.config = config
        
        # 深度相机配置
        self.depth_camera_config = {
            'width': config.get('depth_width', 64),
            'height': config.get('depth_height', 64),
            'fov': config.get('depth_fov', 50.0),  # 与config.py同步
            'near_plane': config.get('depth_near', 0.8),  # 与config.py同步
            'far_plane': config.get('depth_far', 12.0),  # 与config.py同步
        }
        
        # 观察相机配置
        self.observer_camera_config = {
            'follow': config.get('camera_follow', True),
            'target': config.get('camera_target', 'leader'),  # 'leader' or 'formation'
            'distance': config.get('camera_distance', 10.0),
            'yaw': config.get('camera_yaw', 45.0),
            'pitch': config.get('camera_pitch', -30.0)
        }
        
        # 渲染配置
        self.render_config = {
            'formation_lines': config.get('render_formation_lines', True),
            'goal_hint': config.get('render_goal_hint', True),
            'debug_info': config.get('render_debug_info', False)
        }
        
        # 相机对象存储
        self.formation_line_ids = []
        
        # 🔧 新增：固定俯视摄像头
        self.fixed_overhead_camera_id = None
        self.fixed_camera_config = {
            'enabled': config.get('fixed_overhead_camera', False),
            'height': config.get('fixed_camera_height', 15.0),
            'distance': config.get('fixed_camera_distance', 0.0),  # 从正上方看
            'yaw': config.get('fixed_camera_yaw', 0.0),
            'pitch': config.get('fixed_camera_pitch', -90.0),  # 垂直向下
        }
        
        # 🔧 新增：摄像头稳定化状态
        self.last_camera_pos = None
        self.last_camera_yaw = None
        
        # 🔧 新增：缓存projection matrix以提高稳定性（只在初始化时计算一次）
        self._cached_projection_matrix = None
        self._projection_matrix_config = None  # 用于检测配置变化
    
    def _get_cached_projection_matrix(self) -> list:
        """获取缓存的投影矩阵，如果配置改变则重新计算"""
        current_config = (
            self.depth_camera_config['width'],
            self.depth_camera_config['height'],
            self.depth_camera_config['fov'],
            self.depth_camera_config['near_plane'],
            self.depth_camera_config['far_plane']
        )
        
        # 如果配置改变或首次计算，则重新计算投影矩阵
        if self._cached_projection_matrix is None or self._projection_matrix_config != current_config:
            self._cached_projection_matrix = p.computeProjectionMatrixFOV(
                fov=self.depth_camera_config['fov'],
                aspect=self.depth_camera_config['width'] / self.depth_camera_config['height'],
                nearVal=self.depth_camera_config['near_plane'],
                farVal=self.depth_camera_config['far_plane'],
                physicsClientId=self.client
            )
            self._projection_matrix_config = current_config
        
        return self._cached_projection_matrix
    
    def setup_fixed_overhead_camera(self, leader_drone) -> bool:
        """设置固定在领航者无人机上方的平视摄像头 - 跟随无人机移动"""
        if not self.fixed_camera_config['enabled']:
            return False
            
        try:
            # 获取领航者无人机的位置和朝向
            leader_pos, leader_ori = p.getBasePositionAndOrientation(leader_drone.drone, self.client)
            leader_pos = np.array(leader_pos)
            # 计算摄像头位置（在领航者上方固定高度）
            camera_height_offset = self.fixed_camera_config['height']
            camera_pos = leader_pos + np.array([0, 0, camera_height_offset])
            # 获取无人机当前欧拉角
            euler = p.getEulerFromQuaternion(leader_ori)
            camera_yaw = np.degrees(euler[2])  # Z轴为yaw
            camera_pitch = np.degrees(euler[1])  # Y轴为pitch
            # 获取领航者的前进方向（Y轴）
            rot_mat = p.getMatrixFromQuaternion(leader_ori)
            forward_vec = np.array([rot_mat[3], rot_mat[4], rot_mat[5]])  # Y轴方向
            forward_vec = forward_vec / np.linalg.norm(forward_vec)
            # 计算目标点（沿领航者前进方向）
            target_distance = 5.0  # 向前看5米
            target_pos = leader_pos + forward_vec * target_distance
            # 设置摄像头看向领航者前进方向，yaw/pitch用无人机当前欧拉角
            p.resetDebugVisualizerCamera(
                cameraDistance=self.fixed_camera_config['distance'],
                cameraYaw=camera_yaw,
                cameraPitch=camera_pitch,
                cameraTargetPosition=target_pos.tolist(),  # 看向前进方向
                physicsClientId=self.client
            )
            
            return True
            
        except Exception as e:
            print(f"设置领航者上方固定平视摄像头失败: {e}")
            return False
    
    def enable_synthetic_camera_views(self):
        """启用PyBullet的合成相机视图显示"""
        try:
            # 启用GUI面板
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.client)
            
            # 启用RGB缓冲区预览
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1, physicsClientId=self.client)
            
            # 启用深度缓冲区预览
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1, physicsClientId=self.client)
            
            # 启用分割掩码预览
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1, physicsClientId=self.client)
            
            # print("✅ 已启用合成相机视图显示 (RGB、Depth、Segmentation)")  # 注释掉训练时的输出
            return True
        except Exception as e:
            print(f"启用合成相机视图失败: {e}")
            return False
    
    def get_leader_camera_image(self, leader_drone) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取领航者相机图像 - 现在使用URDF camera_link pose"""
        try:
            # 获取URDF camera_link的pose
            cam_pos, cam_orn = leader_drone.get_camera_pose()
            return self.get_leader_camera_image_by_pose(cam_pos, cam_orn)
        except Exception as e:
            print(f"获取领航者相机图像失败: {e}")
            return self._get_default_images()
    
    def get_leader_camera_frame(self, leader_drone) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """获取领航者相机完整帧（RGB, Depth, Segmentation）- 现在使用URDF camera_link pose"""
        try:
            # 获取URDF camera_link的pose
            cam_pos, cam_orn = leader_drone.get_camera_pose()
            return self.get_leader_camera_frame_by_pose(cam_pos, cam_orn)
        except Exception as e:
            print(f"获取领航者相机完整帧失败: {e}")
            rgb, depth = self._get_default_images()
            seg = np.full(depth.shape, -1, dtype=np.int32)
            return rgb, depth, seg
    
    def get_leader_camera_image_by_pose(self, cam_pos, cam_orn) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        通过摄像头的位姿采集RGB和深度图
        Args:
            cam_pos: 摄像头位置 (3,) - 可以是list或tuple
            cam_orn: 摄像头四元数 (4,) - 可以是list或tuple
        Returns:
            rgb_image, depth_image
        """
        try:
            # 确保输入是numpy数组
            cam_pos = np.array(cam_pos)
            cam_orn = np.array(cam_orn)
            
            # 从四元数获取旋转矩阵
            rot_mat = p.getMatrixFromQuaternion(cam_orn)
            rot_mat = np.array(rot_mat).reshape(3, 3)
            
            # 提取无人机的三个主轴
            x_axis = rot_mat[:, 0]  # 第一列是X轴
            y_axis = rot_mat[:, 1]  # 第二列是Y轴
            z_axis = rot_mat[:, 2]  # 第三列是Z轴
            
            # 使用X轴作为前向向量，但对其进行水平化处理
            forward_vec = x_axis.copy()
            
            # 将前向向量投影到水平面上（去除Z轴分量）
            forward_vec_planar = np.array([forward_vec[0], forward_vec[1], 0.0])
            
            # 如果水平投影为零向量（例如相机垂直向上或向下），使用默认前向
            if np.linalg.norm(forward_vec_planar) < 0.01:
                forward_vec_planar = np.array([1.0, 0.0, 0.0])  # 默认向X轴正方向
            
            # 归一化水平前向向量
            forward_vec_planar = forward_vec_planar / np.linalg.norm(forward_vec_planar)
            
            # 平滑混合原始前向和水平化前向 - 90%水平向量 + 10%原始向量，确保视角基本水平
            # 这种混合可以保留轻微的俯仰角但防止视角过度抬高
            forward_vec_final = 0.9 * forward_vec_planar + 0.1 * forward_vec
            forward_vec_final = forward_vec_final / np.linalg.norm(forward_vec_final)
            
            # 计算目标点（相机前方8米）
            target_pos = cam_pos + forward_vec_final * 8.0
            
            # 视图矩阵 - 始终使用全局Z轴作为相机的"上"向量，确保视角不会倾斜
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_pos.tolist(),
                cameraTargetPosition=target_pos.tolist(),
                cameraUpVector=[0, 0, 1]  # 全局Z轴作为上向量
            )
            proj_matrix = self._get_cached_projection_matrix()
            width = self.depth_camera_config['width']
            height = self.depth_camera_config['height']
            images = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.client,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            )
            rgb_buffer = np.array(images[2]).reshape((height, width, 4))
            rgb_image = rgb_buffer[:, :, :3].astype(np.uint8)
            depth_buffer = np.array(images[3]).reshape((height, width))
            depth_image = self._convert_depth_buffer(depth_buffer)
            return rgb_image, depth_image
        except Exception as e:
            print(f"通过camera_link采集相机图像失败: {e}")
            return self._get_default_images()
    """相机管理器 - 负责相机设置、图像获取和渲染"""

    def get_leader_camera_frame_by_pose(self, cam_pos, cam_orn) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        通过摄像头的位姿采集RGB、深度和分割图
        Args:
            cam_pos: 摄像头位置 (3,) - 可以是list或tuple
            cam_orn: 摄像头四元数 (4,) - 可以是list或tuple
        Returns:
            rgb_image, depth_image, seg_mask
        """
        try:
            # 确保输入是numpy数组
            cam_pos = np.array(cam_pos)
            cam_orn = np.array(cam_orn)
            
            # 从四元数获取旋转矩阵
            rot_mat = p.getMatrixFromQuaternion(cam_orn)
            rot_mat = np.array(rot_mat).reshape(3, 3)
            
            # 提取无人机的三个主轴
            x_axis = rot_mat[:, 0]  # 第一列是X轴
            y_axis = rot_mat[:, 1]  # 第二列是Y轴
            z_axis = rot_mat[:, 2]  # 第三列是Z轴
            
            # 使用X轴作为前向向量，但对其进行水平化处理
            forward_vec = x_axis.copy()
            
            # 将前向向量投影到水平面上（去除Z轴分量）
            forward_vec_planar = np.array([forward_vec[0], forward_vec[1], 0.0])
            
            # 如果水平投影为零向量（例如相机垂直向上或向下），使用默认前向
            if np.linalg.norm(forward_vec_planar) < 0.01:
                forward_vec_planar = np.array([1.0, 0.0, 0.0])  # 默认向X轴正方向
            
            # 归一化水平前向向量
            forward_vec_planar = forward_vec_planar / np.linalg.norm(forward_vec_planar)
            
            # 平滑混合原始前向和水平化前向 - 90%水平向量 + 10%原始向量，确保视角基本水平
            # 这种混合可以保留轻微的俯仰角但防止视角过度抬高
            forward_vec_final = 0.9 * forward_vec_planar + 0.1 * forward_vec
            forward_vec_final = forward_vec_final / np.linalg.norm(forward_vec_final)
            
            # 计算目标点（相机前方8米）
            target_pos = cam_pos + forward_vec_final * 8.0
            
            # 视图矩阵 - 始终使用全局Z轴作为相机的"上"向量，确保视角不会倾斜
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_pos.tolist(),
                cameraTargetPosition=target_pos.tolist(),
                cameraUpVector=[0, 0, 1]  # 全局Z轴作为上向量
            )
            proj_matrix = self._get_cached_projection_matrix()
            width = self.depth_camera_config['width']
            height = self.depth_camera_config['height']
            images = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.client,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            )
            rgb_buffer = np.array(images[2]).reshape((height, width, 4))
            rgb_image = rgb_buffer[:, :, :3].astype(np.uint8)
            depth_buffer = np.array(images[3]).reshape((height, width))
            depth_image = self._convert_depth_buffer(depth_buffer)
            seg_mask = np.array(images[4]).reshape((height, width)).astype(np.int32)
            return rgb_image, depth_image, seg_mask
        except Exception as e:
            print(f"通过camera_link采集相机完整帧失败: {e}")
            rgb, depth = self._get_default_images()
            seg = np.full(depth.shape, -1, dtype=np.int32)
            return rgb, depth, seg
    
    def _convert_depth_buffer(self, depth_buffer: np.ndarray) -> np.ndarray:
        """转换深度缓冲区到真实深度值"""
        near = self.depth_camera_config['near_plane']
        far = self.depth_camera_config['far_plane']
        
        # PyBullet深度缓冲区转换
        real_depth = far * near / (far - (far - near) * depth_buffer)
        real_depth = np.clip(real_depth, near, far)
        
        # 统一返回2D数组格式，与state_processor保持一致
        return real_depth.astype(np.float32)
    
    def _get_default_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取默认图像（当相机不可用时）"""
        width = self.depth_camera_config['width']
        height = self.depth_camera_config['height']
        
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        depth_image = np.full((height, width), 5.0, dtype=np.float32)  # 统一为2D格式
        
        return rgb_image, depth_image
    
    def update_debug_camera_for_sidebar(self, leader_drone):
        """更新右侧观察相机 - 显示整个领航者和环境的关系"""
        try:
            # 获取领航者位置
            pos, _ = p.getBasePositionAndOrientation(leader_drone.drone, self.client)

            # 使用观察相机配置来设置右侧相机
            # 显示整个场景和编队关系
            camera_distance = self.observer_camera_config['distance']
            camera_yaw = self.observer_camera_config['yaw']
            camera_pitch = self.observer_camera_config['pitch']

            # 设置观察相机 - 显示整个场景
            p.resetDebugVisualizerCamera(
                cameraDistance=camera_distance,
                cameraYaw=camera_yaw,
                cameraPitch=camera_pitch,
                cameraTargetPosition=pos,  # 看向领航者位置
                physicsClientId=self.client
            )

        except Exception as e:
            print(f"更新右侧观察相机失败: {e}")
    
    def update_synthetic_camera_panel(self, leader_drone):
        """更新左侧Synthetic Camera面板显示领航者深度相机视角，并显示掩码后的深度图像"""
        try:
            # 获取相机位姿
            cam_pos, cam_orn = leader_drone.get_camera_pose()
            
            # 计算相机视角
            cam_pos = np.array(cam_pos)
            cam_orn = np.array(cam_orn)
            
            # 从四元数获取旋转矩阵
            rot_mat = p.getMatrixFromQuaternion(cam_orn)
            rot_mat = np.array(rot_mat).reshape(3, 3)
            
            # 提取X轴作为前向向量并水平化
            forward_vec = rot_mat[:, 0]
            forward_vec_planar = np.array([forward_vec[0], forward_vec[1], 0.0])
            
            # 处理边缘情况
            if np.linalg.norm(forward_vec_planar) < 0.01:
                forward_vec_planar = np.array([1.0, 0.0, 0.0])
            else:
                forward_vec_planar = forward_vec_planar / np.linalg.norm(forward_vec_planar)
            
            # 混合向量，保留一些垂直分量
            forward_vec_final = 0.9 * forward_vec_planar + 0.1 * forward_vec
            forward_vec_final = forward_vec_final / np.linalg.norm(forward_vec_final)
            
            # 计算目标点和视图矩阵
            target_pos = cam_pos + forward_vec_final * 8.0
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_pos.tolist(),
                cameraTargetPosition=target_pos.tolist(),
                cameraUpVector=[0, 0, 1]
            )
            proj_matrix = self._get_cached_projection_matrix()

            # 获取原始相机图像
            rgb, depth, seg = self.get_leader_camera_frame_by_pose(cam_pos, cam_orn)
            
            # 应用分割掩码处理
            leader_body_unique_id = int(leader_drone.drone)
            obj_ids = (seg >> 24).astype(np.int32)
            self_mask = (obj_ids == leader_body_unique_id)
            
            # 添加说明文本
            if hasattr(self, 'debug_depth_text_id'):
                try:
                    p.removeUserDebugItem(self.debug_depth_text_id, physicsClientId=self.client)
                except:
                    pass
            
            self.debug_depth_text_id = p.addUserDebugText(
                "掩码深度图 (无人机自身已过滤)",
                [3, 3, 2.5],  # 位置
                textColorRGB=[1, 1, 1],
                textSize=1.0,
                lifeTime=0.2,
                physicsClientId=self.client
            )
            
            # 调用getCameraImage来更新左侧Synthetic Camera面板
            p.getCameraImage(
                width=self.depth_camera_config['width'],
                height=self.depth_camera_config['height'],
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.client,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            )

            # 返回原始图像和掩码，供其他地方使用
            return rgb, depth, seg, self_mask

        except Exception as e:
            print(f"更新左侧Synthetic Camera面板失败: {e}")
            return None, None, None, None
    
    def update_observer_camera(self, drones: list, leader_index: int = 0, camera_config: dict = None):
        """更新观察相机位置"""
        # 如果提供了新的相机配置，则更新
        if camera_config:
            self.observer_camera_config.update(camera_config)
        
        if not self.observer_camera_config['follow'] or not drones:
            return
        
        try:
            # 确定相机目标
            target = self.observer_camera_config['target']
            if target == "leader":
                target_pos, _ = p.getBasePositionAndOrientation(
                    drones[leader_index].drone, self.client
                )
                target_pos = np.array(target_pos)
                # 使用固定的yaw/pitch，不跟随无人机旋转
                yaw = self.observer_camera_config['yaw']
                pitch = self.observer_camera_config['pitch']
            elif target == "formation":
                # 计算编队中心
                positions = []
                for drone in drones:
                    pos, _ = p.getBasePositionAndOrientation(drone.drone, self.client)
                    positions.append(pos)
                target_pos = np.mean(positions, axis=0)
                # 编队中心无统一朝向，使用默认yaw/pitch
                yaw = self.observer_camera_config['yaw']
                pitch = self.observer_camera_config['pitch']
            else:
                return
            distance = self.observer_camera_config['distance']
            # 设置相机
            p.resetDebugVisualizerCamera(
                cameraDistance=distance,
                cameraYaw=yaw,
                cameraPitch=pitch,
                cameraTargetPosition=target_pos,
                physicsClientId=self.client
            )
            
        except Exception as e:
            print(f"更新观察相机失败: {e}")
    
    def render_formation_lines(self, drones: list, leader_index: int = 0):
        """渲染编队连线"""
        if not self.render_config['formation_lines'] or len(drones) < 2:
            return
        
        # 不需要清理旧线条，使用lifeTime自动过期
        
        try:
            # 获取所有无人机位置
            positions = []
            for drone in drones:
                pos, _ = p.getBasePositionAndOrientation(drone.drone, self.client)
                positions.append(np.array(pos))
            
            leader_pos = positions[leader_index]
            
            # 绘制领航者到跟随者的连线
            for i, follower_pos in enumerate(positions):
                if i != leader_index:
                    line_id = p.addUserDebugLine(
                        leader_pos, follower_pos,
                        lineColorRGB=[0.0, 1.0, 0.0],
                        lineWidth=1.5,
                        lifeTime=0.05,  # 极短生命周期，避免累积
                        physicsClientId=self.client
                    )
                    # 不需要存储line_id，让它自动过期
                    
        except Exception as e:
            print(f"渲染编队连线失败: {e}")
    
    def render_goal_hint(self, drones: list, goal: np.ndarray, leader_index: int = 0):
        """渲染目标提示"""
        if not self.render_config['goal_hint'] or not drones or goal is None:
            return
        
        try:
            leader_pos, _ = p.getBasePositionAndOrientation(
                drones[leader_index].drone, self.client
            )
            leader_pos = np.array(leader_pos)
            
            # 绘制领航者到目标的连线
            line_id = p.addUserDebugLine(
                leader_pos, goal,
                lineColorRGB=[0.2, 0.8, 1.0],
                lineWidth=2.0,
                lifeTime=0.05,  # 极短生命周期，避免累积
                physicsClientId=self.client
            )
            
            # 显示距离文本
            distance = np.linalg.norm(goal - leader_pos)
            text_id = p.addUserDebugText(
                f"d={distance:.1f}m",
                goal + np.array([0, 0, 0.8]),
                textColorRGB=[0.2, 0.8, 1.0],
                textSize=1.1,
                lifeTime=0.05,  # 极短生命周期，避免累积
                physicsClientId=self.client
            )
            
        except Exception as e:
            print(f"渲染目标提示失败: {e}")
    
    def render_debug_info(self, info: Dict[str, Any], position: np.ndarray):
        """渲染调试信息"""
        if not self.render_config['debug_info']:
            return
        
        try:
            debug_text = []
            for key, value in info.items():
                if isinstance(value, float):
                    debug_text.append(f"{key}: {value:.2f}")
                else:
                    debug_text.append(f"{key}: {value}")
            
            text_content = "\\n".join(debug_text)
            text_id = p.addUserDebugText(
                text_content,
                position + np.array([0, 0, 2.0]),
                textColorRGB=[1.0, 1.0, 0.0],
                textSize=0.8,
                lifeTime=0.5,  # 0.5秒生命周期，避免累积
                physicsClientId=self.client
            )
            self.formation_line_ids.append(text_id)
            
        except Exception as e:
            print(f"渲染调试信息失败: {e}")
    
    def cleanup(self):
        """清理相机资源"""
        # 清理调试线条
        for line_id in self.formation_line_ids:
            try:
                p.removeUserDebugItem(line_id, physicsClientId=self.client)
            except:
                pass
        self.formation_line_ids.clear()
        
        # 清理深度图像文本
        if hasattr(self, 'debug_depth_text_id'):
            try:
                p.removeUserDebugItem(self.debug_depth_text_id, physicsClientId=self.client)
            except:
                pass


def create_default_camera_config() -> Dict[str, Any]:
    """创建优化的默认相机配置

    相机配置层次：
    1. 深度相机：领航者无人机身上的物理相机，用于获取深度图像（位置和朝向由URDF camera_link定义）
    2. 观察相机：PyBullet GUI显示相机，用于可视化
    3. 固定俯视相机：可选的固定视角相机，位于领航者上方
    4. 渲染配置：控制可视化元素的显示
    """
    return {
        # === 深度相机配置（领航者无人机身上的物理相机） ===
        # 注意：相机位置和朝向现在由URDF中的camera_link定义，不再需要手动配置
        'depth_width': 64,
        'depth_height': 64,
        'depth_fov': 50.0,  # 稍微减少FOV获得更集中的视野
        'depth_near': 0.8,  # 与config.py同步，避免渲染机身
        'depth_far': 12.0,  # 与config.py同步，看得更远

        # === 观察相机配置（PyBullet GUI显示相机） ===
        'camera_follow': True,  # 是否跟随无人机
        'camera_target': 'leader',  # 相机目标：'leader'（领航者）、'formation'（编队中心）
        'camera_distance': 3.0,  # 减少距离，更近距离观察领航者
        'camera_yaw': 60.0,  # 减少偏航角，更直接观察领航者
        'camera_pitch': -45.0,  # 调整俯仰角，更好地观察领航者和编队

        # === 固定俯视相机配置（可选的固定视角） ===
        'fixed_overhead_camera': False,  # 是否启用固定俯视摄像头
        'fixed_camera_height': 0.5,      # 固定摄像头在领航者上方的距离
        'fixed_camera_distance': 0.0,    # 摄像头距离（0表示正上方）
        'fixed_camera_yaw': 0.0,         # 固定摄像头偏航角
        'fixed_camera_pitch': 1.5,       # 固定摄像头俯仰角（1.5度向上看）

        # === 渲染配置 ===
        'render_formation_lines': True,  # 是否渲染编队线
        'render_goal_hint': True,  # 是否给领航者绘制到目标的提示线/文字
        'render_debug_info': False,  # 是否渲染调试信息
    }