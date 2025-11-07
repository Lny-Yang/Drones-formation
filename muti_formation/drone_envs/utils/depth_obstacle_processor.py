"""
深度图像处理和避障特征提取模块
Inspired by "Towards Monocular Vision Based Collision Avoidance Using Deep Reinforcement Learning"
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

class DepthObstacleProcessor:
    """深度图像处理和避障特征提取类"""
    
    def __init__(self, 
                 depth_image_size: Tuple[int, int] = (128, 160),
                 collision_threshold: float = 0.2,
                 depth_scale: float = 4.0,
                 max_depth: float = 2.0,
                 cnn_feature_dim: int = 128):
        """
        初始化深度处理器
        
        Args:
            depth_image_size: 深度图像尺寸 (height, width)
            collision_threshold: 碰撞检测阈值 (米)
            depth_scale: 深度值缩放因子
            max_depth: 最大深度值
            cnn_feature_dim: CNN特征维度
        """
        self.height, self.width = depth_image_size
        self.collision_threshold = collision_threshold
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.cnn_feature_dim = cnn_feature_dim
        
        # 初始化CNN模型
        self.cnn_model = self._build_cnn_model()
        self.cnn_model.eval()  # 设置为评估模式
        
    def _build_cnn_model(self) -> nn.Module:
        """构建高效的CNN深度特征提取模型"""
        class EfficientDepthCNN(nn.Module):
            def __init__(self, input_channels=1, feature_dim=128):
                super(EfficientDepthCNN, self).__init__()
                
                # 卷积层定义
                self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)  # 128x160 -> 64x80
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 64x80 -> 32x40
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 32x40 -> 16x20
                self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 16x20 -> 8x10
                
                # 自适应平均池化到固定大小
                self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 8x10 -> 4x4
                
                # 全连接层
                self.fc = nn.Linear(256 * 4 * 4, feature_dim)
                
                # 批归一化
                self.bn1 = nn.BatchNorm2d(32)
                self.bn2 = nn.BatchNorm2d(64)
                self.bn3 = nn.BatchNorm2d(128)
                self.bn4 = nn.BatchNorm2d(256)
                
            def forward(self, x):
                # 卷积块1
                x = F.relu(self.bn1(self.conv1(x)))
                # 卷积块2
                x = F.relu(self.bn2(self.conv2(x)))
                # 卷积块3
                x = F.relu(self.bn3(self.conv3(x)))
                # 卷积块4
                x = F.relu(self.bn4(self.conv4(x)))
                
                # 自适应池化
                x = self.adaptive_pool(x)
                
                # 展平
                x = x.view(x.size(0), -1)
                
                # 全连接层
                x = self.fc(x)
                return x
        
        return EfficientDepthCNN(input_channels=1, feature_dim=self.cnn_feature_dim)
        
    def preprocess_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        """
        预处理深度图像
        
        Args:
            depth_image: 原始深度图像 (H, W) 或 (H, W, 1)
            
        Returns:
            处理后的深度图像
        """
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        
        # 处理NaN值和无穷值
        depth_image = np.array(depth_image, dtype=np.float32)
        depth_image[np.isnan(depth_image)] = 5.0
        depth_image[np.isinf(depth_image)] = 5.0
        
        # 归一化处理
        depth_image = depth_image / self.depth_scale
        depth_image = np.clip(depth_image, 0.0, self.max_depth)
        
        return depth_image
        
    def extract_cnn_features(self, depth_map: np.ndarray) -> List[float]:
        """
        使用CNN提取深度图像特征
        
        Args:
            depth_map: 预处理后的深度图像
            
        Returns:
            CNN特征向量
        """
        # 确保输入格式正确
        if len(depth_map.shape) == 2:
            depth_map = depth_map[np.newaxis, np.newaxis, :, :]  # 添加batch和channel维度
        elif len(depth_map.shape) == 3:
            depth_map = depth_map[np.newaxis, :, :, :]  # 添加batch维度
        else:
            raise ValueError(f"Unsupported depth_map shape: {depth_map.shape}")
        
        # 转换为torch tensor
        depth_tensor = torch.from_numpy(depth_map).float()
        
        # 前向传播
        with torch.no_grad():
            features = self.cnn_model(depth_tensor)
            features = features.squeeze(0).cpu().numpy()
        
        # 防御性编程：检查并处理异常值
        features = np.array(features, dtype=np.float32)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"警告: CNN特征提取出现NaN或Inf值，使用零向量替代")
            features = np.zeros(self.cnn_feature_dim, dtype=np.float32)
        
        # 截断极端值
        features = np.clip(features, -10.0, 10.0)
        
        return features.tolist()
    
    def extract_depth_features(self, depth_map: np.ndarray) -> List[float]:
        """
        提取深度特征（使用CNN）
        
        Args:
            depth_map: 预处理后的深度图像
            
        Returns:
            特征向量
        """
        return self.extract_cnn_features(depth_map)
    
    def detect_obstacles(self, depth_map: np.ndarray) -> Tuple[bool, float]:
        """
        障碍物检测 - 优化以更好检测前方开放空间
        
        Args:
            depth_map: 预处理后的深度图像
            
        Returns:
            (是否检测到障碍物, 最小深度距离)
        """
        # 提取中心区域进行障碍物检测
        h, w = depth_map.shape
        
        # 区分前方和侧面区域
        forward_region = depth_map[h//3:2*h//3, w//3:2*w//3]  # 中心前方区域
        full_detection_region = depth_map[h//4:3*h//4, w//4:3*w//4]  # 整体检测区域
        
        # 计算最小深度距离
        min_depth_forward = np.min(forward_region) if forward_region.size > 0 else float('inf')
        min_depth_full = np.min(full_detection_region) if full_detection_region.size > 0 else float('inf')
        
        # 使用前方区域的最小深度，除非它是无穷大
        min_depth = min_depth_forward if min_depth_forward < float('inf') else min_depth_full
        
        # 二值化处理检测障碍物密度 - 使用配置的碰撞阈值
        obstacle_threshold = self.collision_threshold  # 使用配置的碰撞阈值而不是硬编码
        binary_mask = (full_detection_region < obstacle_threshold).astype(np.float32)
        obstacle_density = np.mean(binary_mask)
        
        # 专门检查前方是否开放 - 如果前方区域最小深度大于阈值，认为前方开放
        forward_is_open = min_depth_forward > obstacle_threshold
        
        # 障碍物检测逻辑 - 调整密度阈值
        obstacle_detected = obstacle_density > 0.1  # 提高阈值，减少误检
            
        return obstacle_detected, float(min_depth)
    
    def get_obstacle_analysis(self, depth_map: np.ndarray) -> Dict[str, float]:
        """
        🔧 重构：只做障碍物分析，不计算奖励
        
        职责：提供客观的障碍物信息，供RewardCalculator使用
        - 障碍物检测（是/否）
        - 最小距离（标准化）
        - 前方开放度（0-1）
        - 物理最小距离（米）
        
        Args:
            depth_map: 预处理后的深度图像
            
        Returns:
            障碍物分析信息字典
        """
        obstacle_detected, min_depth = self.detect_obstacles(depth_map)
        
        # 计算物理距离
        physical_min_depth = min_depth * self.depth_scale
        
        # 计算前方开放度（0-1）
        h, w = depth_map.shape
        forward_region = depth_map[h//3:2*h//3, w//3:2*w//3]
        forward_openness = np.mean(forward_region * self.depth_scale > 2.0)
        
        # 计算危险等级（0-1，1最危险）
        # 🔥 优化：根据最大速度5m/s调整距离阈值
        # 考虑反应时间：0.5秒@5m/s = 2.5米，1秒@5m/s = 5米
        danger_level = 0.0
        if physical_min_depth < 1.0:  # <1米，非常危险（0.2秒反应时间）
            danger_level = 1.0
        elif physical_min_depth < 2.0:  # <2米，危险（0.4秒反应时间）
            danger_level = 0.7
        elif physical_min_depth < 3.0:  # <3米，需注意（0.6秒反应时间）
            danger_level = 0.4
        elif physical_min_depth < 4.0:  # <4米，稍有风险（0.8秒反应时间）
            danger_level = 0.2
        
        return {
            'obstacle_detected': bool(obstacle_detected),
            'min_depth': float(min_depth),  # 标准化深度 [0-2.0]
            'physical_min_depth': float(physical_min_depth),  # 物理距离（米）
            'forward_openness': float(forward_openness),  # 前方开放度 [0-1]
            'danger_level': float(danger_level),  # 危险等级 [0-1]
            'is_imminent_collision': physical_min_depth <= self.collision_threshold * self.depth_scale
        }
        
    def get_navigation_features(self, depth_map: np.ndarray) -> np.ndarray:
        """
        获取导航相关的深度特征
        
        Args:
            depth_map: 深度图像
            
        Returns:
            特征向量
        """
        # 使用统一的特征提取接口
        depth_features = self.extract_depth_features(depth_map)
        
        # 障碍物检测特征
        obstacle_detected, min_depth = self.detect_obstacles(depth_map)
        
        # 组合特征
        navigation_features = depth_features + [
            float(obstacle_detected),
            min_depth,
        ]
        
        return np.array(navigation_features, dtype=np.float32)
