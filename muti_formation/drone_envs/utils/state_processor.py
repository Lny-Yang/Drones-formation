"""
多无人机编队的状态处理模块
"""
import numpy as np
import pybullet as p
from typing import Dict, Any, Tuple, List, Optional
from .depth_obstacle_processor import DepthObstacleProcessor

# 统一读取环境配置，保持深度缩放等参数一致
try:
    from ..config import multi_drone_env as env_config
except ImportError:
    from drone_envs.config import multi_drone_env as env_config


class StateProcessor:
    """状态处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化状态处理器
        
        Args:
            config: 状态配置参数
        """
        self.config = config
        
        # 深度处理器
        self.depth_processor = DepthObstacleProcessor(
            depth_image_size=(config.get('depth_height', 64), config.get('depth_width', 64)),
            collision_threshold=config.get('collision_threshold', env_config.get('collision_distance', 0.8) / env_config.get('depth_scale', 7.5)),
            depth_scale=config.get('depth_scale', env_config.get('depth_scale', 7.5)),
            max_depth=config.get('max_depth', env_config.get('max_depth', 2.0)),
            cnn_feature_dim=config.get('cnn_feature_dim', env_config.get('cnn_feature_dim', 256))
        )
        
        # 状态维度配置
        self.position_dim = 3  # x, y, z
        self.velocity_dim = 3  # vx, vy, vz
        self.orientation_dim = 4  # quaternion
        self.target_dim = 3  # target position
        
        # CNN特征维度 + 增强特征维度
        self.depth_features_dim = config.get('depth_feature_dim', 130)
        
        # 计算总状态维度
        self.state_dim = (self.position_dim + self.velocity_dim + 
                         self.orientation_dim + self.target_dim + 
                         self.depth_features_dim)
        
        # 相机配置
        self.camera_config = {
            'width': config.get('depth_width', env_config.get('depth_width', 64)),
            'height': config.get('depth_height', env_config.get('depth_height', 64)),
            'fov': config.get('camera_fov', env_config.get('depth_fov', 70.0)),
            'near_plane': config.get('depth_near', env_config.get('depth_near', 0.3)),
            'far_plane': config.get('depth_far', env_config.get('depth_far', 15.0))
        }
        
    def get_state_dimension(self) -> int:
        """获取状态维度"""
        return self.state_dim
    
    def build_state(self, 
                   drone_id: int,
                   position: np.ndarray,
                   velocity: np.ndarray,
                   orientation: np.ndarray,
                   target_position: np.ndarray,
                   depth_image: Optional[np.ndarray] = None,
                   enforce_planar: bool = False) -> np.ndarray:
        """
        构建状态向量
        
        Args:
            drone_id: 无人机ID
            position: 位置 [x, y, z]
            velocity: 速度 [vx, vy, vz]
            orientation: 四元数 [x, y, z, w]
            target_position: 目标位置 [x, y, z]
            depth_image: 深度图像
            enforce_planar: 是否强制平面模式（不包含z轴信息）
            
        Returns:
            状态向量
        """
        state_components = []
        
        # 1. 位置信息（归一化）
        if enforce_planar:
            # 平面模式：只使用x, y位置
            normalized_position = self._normalize_position_planar(position)
        else:
            normalized_position = self._normalize_position(position)
        state_components.extend(normalized_position)
        
        # 2. 速度信息（归一化）
        if enforce_planar:
            # 平面模式：只使用x, y速度
            normalized_velocity = self._normalize_velocity_planar(velocity)
        else:
            normalized_velocity = self._normalize_velocity(velocity)
        state_components.extend(normalized_velocity)
        
        # 3. 朝向信息（四元数）
        state_components.extend(orientation)
        
        # 4. 目标相对位置（归一化）
        if enforce_planar:
            # 平面模式：只使用x, y相对位置
            relative_target = self._compute_relative_target_planar(position, target_position)
        else:
            relative_target = self._compute_relative_target(position, target_position)
        state_components.extend(relative_target)
        
        # 5. 深度特征
        if depth_image is not None:
            depth_features = self._extract_depth_features(depth_image)
        else:
            depth_features = np.zeros(self.depth_features_dim)
        state_components.extend(depth_features)
        
        return np.array(state_components, dtype=np.float32)
    
    def capture_depth_image(self, drone_id: int, position: np.ndarray, orientation: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        捕获深度图像并提取信息
        
        Args:
            drone_id: 无人机ID
            position: 位置
            orientation: 朝向（四元数）
            
        Returns:
            (深度图像, 深度信息字典)
        """
        # 将四元数转换为旋转矩阵
        rotation_matrix = p.getMatrixFromQuaternion(orientation)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
        
        # 计算相机位置和朝向
        camera_position = position + np.array([0, 0, 0.1])  # 稍微向上偏移
        
        # 计算目标点（相机朝向）- 使用Y轴作为前向，与CameraManager保持一致
        forward_vector = rotation_matrix[:, 1]  # Y轴为前向
        target_position = camera_position + forward_vector * 2.0
        
        # 计算上向量
        up_vector = rotation_matrix[:, 2]  # Z轴为上向
        
        # 计算视图矩阵
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=up_vector
        )
        
        # 计算投影矩阵
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_config['fov'],
            aspect=self.camera_config['width'] / self.camera_config['height'],
            nearPlane=self.camera_config['near_plane'],
            farPlane=self.camera_config['far_plane']
        )
        
        # 渲染深度图像
        width = self.camera_config['width']
        height = self.camera_config['height']
        
        _, _, _, depth_buffer, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        # 转换深度缓冲区
        depth_image = self._convert_depth_buffer(depth_buffer)
        
        # 处理深度图像
        processed_depth = self.depth_processor.preprocess_depth_image(depth_image)
        
        # 提取深度信息
        depth_info = self._extract_depth_info(processed_depth)
        
        return processed_depth, depth_info
    
    def _normalize_position_planar(self, position: np.ndarray) -> List[float]:
        """归一化位置信息 - 平面模式（只使用x, y）"""
        # 30米房间边界：[-15, 15] x [-15, 15]
        normalized = [
            np.clip(position[0] / 15.0, -1.0, 1.0),
            np.clip(position[1] / 15.0, -1.0, 1.0)
        ]
        return normalized
    
    def _normalize_velocity_planar(self, velocity: np.ndarray) -> List[float]:
        """归一化速度信息 - 平面模式（只使用x, y）"""
        # 假设最大速度为 5 m/s
        max_velocity = 5.0
        normalized = [
            np.clip(velocity[0] / max_velocity, -1.0, 1.0),
            np.clip(velocity[1] / max_velocity, -1.0, 1.0)
        ]
        return normalized
    
    def _compute_relative_target_planar(self, position: np.ndarray, target_position: np.ndarray) -> List[float]:
        """计算相对目标位置 - 平面模式（只使用x, y）
        
        🔧 关键修复：直接返回相对位置，不归一化！
        - 让PPO能感知距离信息
        - 归一化会在observation_manager中统一处理
        """
        relative = target_position[:2] - position[:2]  # 只使用x, y分量
        
        # 直接返回相对位置（米），不做归一化
        # ObservationManager会统一归一化到[-15, 15]范围
        return [relative[0], relative[1]]
    
    def _normalize_velocity(self, velocity: np.ndarray) -> List[float]:
        """归一化速度信息"""
        # 假设最大速度为 5 m/s
        max_velocity = 5.0
        normalized = [
            np.clip(velocity[0] / max_velocity, -1.0, 1.0),
            np.clip(velocity[1] / max_velocity, -1.0, 1.0),
            np.clip(velocity[2] / max_velocity, -1.0, 1.0)
        ]
        return normalized
    
    def _compute_relative_target(self, position: np.ndarray, target_position: np.ndarray) -> List[float]:
        """计算相对目标位置 - 3D模式
        
        🔧 关键修复：直接返回相对位置，不归一化！
        - 让PPO能感知距离信息
        - 归一化会在observation_manager中统一处理
        """
        relative = target_position - position
        
        # 直接返回相对位置（米），不做归一化
        # ObservationManager会统一归一化到[-15, 15]范围
        return [relative[0], relative[1], relative[2]]
    
    def _extract_depth_features(self, depth_image: np.ndarray) -> List[float]:
        """提取深度特征"""
        if depth_image is None or depth_image.size == 0:
            return [1.0] * self.depth_features_dim
            
        # 确保深度图像是2D的
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        elif len(depth_image.shape) == 1:
            # 如果是1D数组，说明形状有问题，返回默认值
            return [1.0] * self.depth_features_dim
        
        # 重要：先预处理深度图像再提取特征
        preprocessed_depth = self.depth_processor.preprocess_depth_image(depth_image)
        return self.depth_processor.get_navigation_features(preprocessed_depth).tolist()
    
    def _convert_depth_buffer(self, depth_buffer: np.ndarray) -> np.ndarray:
        """转换深度缓冲区为实际深度值"""
        near = self.camera_config['near_plane']
        far = self.camera_config['far_plane']
        
        # 转换深度缓冲区值到实际深度
        depth_image = far * near / (far - (far - near) * depth_buffer)
        
        return depth_image.astype(np.float32)
    
    def _extract_depth_info(self, depth_image: np.ndarray) -> Dict[str, float]:
        """提取深度信息"""
        # 计算基础深度统计
        h, w = depth_image.shape
        center_region = depth_image[h//4:3*h//4, w//4:3*w//4]
        
        depth_info = {
            'min_depth': float(np.min(center_region)),
            'mean_depth': float(np.mean(center_region)),
            'max_depth': float(np.max(center_region)),
            'std_depth': float(np.std(center_region))
        }
        
        # 区域分析
        forward_region = depth_image[h//3:2*h//3, w//3:2*w//3]
        left_region = depth_image[h//4:3*h//4, :w//3]
        right_region = depth_image[h//4:3*h//4, 2*w//3:]
        
        depth_info.update({
            'forward_min': float(np.min(forward_region)),
            'left_min': float(np.min(left_region)),
            'right_min': float(np.min(right_region))
        })
        
        return depth_info


def create_default_state_config() -> Dict[str, Any]:
    """创建默认状态配置"""
    return {
        'depth_height': env_config.get('depth_height', 64),
        'depth_width': env_config.get('depth_width', 64),
        'collision_threshold': env_config.get('collision_distance', 0.8) / env_config.get('depth_scale', 7.5),
        'depth_scale': env_config.get('depth_scale', 7.5),
        'max_depth': env_config.get('max_depth', 2.0),
        'camera_fov': env_config.get('depth_fov', 70.0),
        'depth_near': env_config.get('depth_near', 0.3),
        'depth_far': env_config.get('depth_far', 15.0),
        'cnn_feature_dim': env_config.get('cnn_feature_dim', 256),
        'depth_feature_dim': env_config.get('depth_feature_dim', 130)
    }