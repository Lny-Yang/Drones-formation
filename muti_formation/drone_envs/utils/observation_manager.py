"""
观测空间管理模块
"""
import numpy as np
import gym
from typing import Dict, Any, List, Tuple


class ObservationSpaceManager:
    """观测空间管理器 - 负责观测空间定义和验证"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化观测空间管理器
        
        Args:
            config: 观测空间配置参数
        """
        self.config = config
        self.config = config
        
        # 基础观测维度配置
        enforce_planar = config.get('enforce_planar', False)
        self.obs_dims = {
            'position': 2 if enforce_planar else config.get('position_dim', 3),      # x, y (平面) 或 x, y, z
            'velocity': 2 if enforce_planar else config.get('velocity_dim', 3),      # vx, vy (平面) 或 vx, vy, vz  
            'orientation': config.get('orientation_dim', 4), # quaternion
            'target': 2 if enforce_planar else config.get('target_dim', 3),          # relative target x,y (平面) 或 x,y,z
            'depth_features': config.get('depth_features_dim', 19), # depth features
        }
        
        # 计算单个智能体的观测维度
        self.single_agent_obs_dim = sum(self.obs_dims.values())
        
        # 多智能体配置
        self.num_agents = config.get('num_agents', 5)
        self.total_obs_dim = self.single_agent_obs_dim * self.num_agents
        
        # 观测值范围配置
        self.obs_bounds = {
            'position_bounds': config.get('position_bounds', [-15.0, 15.0]),  # 30米房间：±15米
            'velocity_bounds': config.get('velocity_bounds', [-5.0, 5.0]),
            'orientation_bounds': config.get('orientation_bounds', [-1.0, 1.0]),
            'target_bounds': config.get('target_bounds', [-30.0, 30.0]),  # 目标可能在房间任意两点间
            'depth_bounds': config.get('depth_bounds', [0.0, 10.0])
        }
        
        # 创建观测空间
        self.observation_space = self._create_observation_space()
        
    def _create_observation_space(self) -> gym.Space:
        """创建观测空间"""
        # 创建观测值的上下界
        low_bounds = []
        high_bounds = []
        
        # 为每个智能体创建边界
        for _ in range(self.num_agents):
            agent_low = []
            agent_high = []
            
        # 为每个智能体创建边界
        for _ in range(self.num_agents):
            agent_low = []
            agent_high = []
            
            # 位置边界
            pos_low, pos_high = self.obs_bounds['position_bounds']
            agent_low.extend([pos_low] * self.obs_dims['position'])
            agent_high.extend([pos_high] * self.obs_dims['position'])
            
            # 速度边界
            vel_low, vel_high = self.obs_bounds['velocity_bounds']
            agent_low.extend([vel_low] * self.obs_dims['velocity'])
            agent_high.extend([vel_high] * self.obs_dims['velocity'])
            
            # 朝向边界
            ori_low, ori_high = self.obs_bounds['orientation_bounds']
            agent_low.extend([ori_low] * self.obs_dims['orientation'])
            agent_high.extend([ori_high] * self.obs_dims['orientation'])
            
            # 目标边界 - 平面模式使用更小的边界
            if self.obs_dims['target'] == 2:  # 平面模式
                tgt_low, tgt_high = -15.0, 15.0  # 房间大小
            else:
                tgt_low, tgt_high = self.obs_bounds['target_bounds']
            agent_low.extend([tgt_low] * self.obs_dims['target'])
            agent_high.extend([tgt_high] * self.obs_dims['target'])
            
            # 深度特征边界
            depth_low, depth_high = self.obs_bounds['depth_bounds']
            agent_low.extend([depth_low] * self.obs_dims['depth_features'])
            agent_high.extend([depth_high] * self.obs_dims['depth_features'])
            
            low_bounds.extend(agent_low)
            high_bounds.extend(agent_high)
        
        # 创建Box空间
        return gym.spaces.Box(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            dtype=np.float32
        )
    
    def get_observation_space(self) -> gym.Space:
        """获取观测空间"""
        return self.observation_space
    
    def get_single_agent_obs_dim(self) -> int:
        """获取单个智能体的观测维度"""
        return self.single_agent_obs_dim
    
    def get_total_obs_dim(self) -> int:
        """获取总观测维度"""
        return self.total_obs_dim
    
    def validate_observation(self, observation: np.ndarray) -> Tuple[bool, str]:
        """验证观测值是否有效"""
        try:
            # 检查维度
            if observation.shape[0] != self.total_obs_dim:
                return False, f"观测维度不匹配: 期望{self.total_obs_dim}, 实际{observation.shape[0]}"
            
            # 检查数值范围
            if not self.observation_space.contains(observation):
                return False, "观测值超出定义范围"
            
            # 检查是否包含NaN或无穷值
            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                return False, "观测值包含NaN或无穷值"
            
            return True, "观测值有效"
            
        except Exception as e:
            return False, f"观测验证失败: {e}"
    
    def split_observation(self, observation: np.ndarray) -> List[np.ndarray]:
        """将联合观测拆分为单个智能体观测"""
        if observation.shape[0] != self.total_obs_dim:
            raise ValueError(f"观测维度不匹配: 期望{self.total_obs_dim}, 实际{observation.shape[0]}")
        
        agent_observations = []
        for i in range(self.num_agents):
            start_idx = i * self.single_agent_obs_dim
            end_idx = start_idx + self.single_agent_obs_dim
            agent_obs = observation[start_idx:end_idx]
            agent_observations.append(agent_obs)
        
        return agent_observations
    
    def combine_observations(self, agent_observations: List[np.ndarray]) -> np.ndarray:
        """将单个智能体观测合并为联合观测"""
        if len(agent_observations) != self.num_agents:
            raise ValueError(f"智能体数量不匹配: 期望{self.num_agents}, 实际{len(agent_observations)}")
        
        # 验证每个智能体观测的维度
        for i, obs in enumerate(agent_observations):
            if obs.shape[0] != self.single_agent_obs_dim:
                raise ValueError(f"智能体{i}观测维度不匹配: 期望{self.single_agent_obs_dim}, 实际{obs.shape[0]}")
        
        return np.concatenate(agent_observations).astype(np.float32)
    
    def parse_single_observation(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        """解析单个智能体的观测"""
        if observation.shape[0] != self.single_agent_obs_dim:
            raise ValueError(f"观测维度不匹配: 期望{self.single_agent_obs_dim}, 实际{observation.shape[0]}")
        
        parsed = {}
        idx = 0
        
        # 解析位置
        parsed['position'] = observation[idx:idx + self.obs_dims['position']]
        idx += self.obs_dims['position']
        
        # 解析速度
        parsed['velocity'] = observation[idx:idx + self.obs_dims['velocity']]
        idx += self.obs_dims['velocity']
        
        # 解析朝向
        parsed['orientation'] = observation[idx:idx + self.obs_dims['orientation']]
        idx += self.obs_dims['orientation']
        
        # 解析目标
        parsed['target'] = observation[idx:idx + self.obs_dims['target']]
        idx += self.obs_dims['target']
        
        # 解析深度特征
        parsed['depth_features'] = observation[idx:idx + self.obs_dims['depth_features']]
        idx += self.obs_dims['depth_features']
        
        return parsed
    
    def get_observation_info(self) -> Dict[str, Any]:
        """获取观测空间信息"""
        return {
            'single_agent_obs_dim': self.single_agent_obs_dim,
            'total_obs_dim': self.total_obs_dim,
            'num_agents': self.num_agents,
            'obs_component_dims': self.obs_dims.copy(),
            'obs_bounds': self.obs_bounds.copy(),
            'space_shape': self.observation_space.shape,
            'space_dtype': self.observation_space.dtype
        }
    
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """归一化观测值到[-1, 1]范围"""
        normalized = np.zeros_like(observation)
        
        for i in range(self.num_agents):
            start_idx = i * self.single_agent_obs_dim
            agent_obs = observation[start_idx:start_idx + self.single_agent_obs_dim]
            
            idx = 0
            # 归一化位置
            pos_low, pos_high = self.obs_bounds['position_bounds']
            pos = agent_obs[idx:idx + self.obs_dims['position']]
            normalized[start_idx + idx:start_idx + idx + self.obs_dims['position']] = \
                2.0 * (pos - pos_low) / (pos_high - pos_low) - 1.0
            idx += self.obs_dims['position']
            
            # 归一化速度
            vel_low, vel_high = self.obs_bounds['velocity_bounds']
            vel = agent_obs[idx:idx + self.obs_dims['velocity']]
            normalized[start_idx + idx:start_idx + idx + self.obs_dims['velocity']] = \
                2.0 * (vel - vel_low) / (vel_high - vel_low) - 1.0
            idx += self.obs_dims['velocity']
            
            # 朝向已经是归一化的
            ori = agent_obs[idx:idx + self.obs_dims['orientation']]
            normalized[start_idx + idx:start_idx + idx + self.obs_dims['orientation']] = ori
            idx += self.obs_dims['orientation']
            
            # 归一化目标
            tgt_low, tgt_high = self.obs_bounds['target_bounds']
            tgt = agent_obs[idx:idx + self.obs_dims['target']]
            normalized[start_idx + idx:start_idx + idx + self.obs_dims['target']] = \
                2.0 * (tgt - tgt_low) / (tgt_high - tgt_low) - 1.0
            idx += self.obs_dims['target']
            
            # 归一化深度特征
            depth_low, depth_high = self.obs_bounds['depth_bounds']
            depth = agent_obs[idx:idx + self.obs_dims['depth_features']]
            normalized[start_idx + idx:start_idx + idx + self.obs_dims['depth_features']] = \
                2.0 * (depth - depth_low) / (depth_high - depth_low) - 1.0
            idx += self.obs_dims['depth_features']
        
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def create_default_observation_config(num_agents: int = 5, depth_features_dim: int = 19, 
                                    use_cnn_features: bool = False, cnn_feature_dim: int = 128,
                                    enforce_planar: bool = False) -> Dict[str, Any]:
    """创建默认观测空间配置"""
    # 从配置文件获取深度特征维度
    from ..config import multi_drone_env as config
    actual_depth_features_dim = config.get('depth_feature_dim', 130)
    
    return {
        'num_agents': num_agents,
        'position_dim': 2 if enforce_planar else 3,  # 平面模式只使用x,y
        'velocity_dim': 2 if enforce_planar else 3,  # 平面模式只使用vx,vy
        'orientation_dim': 4,
        'target_dim': 2 if enforce_planar else 3,    # 平面模式只使用相对x,y
        'depth_features_dim': actual_depth_features_dim,
        'enforce_planar': enforce_planar,  # 添加平面模式标志
        'position_bounds': [-15.0, 15.0],  # 30米房间：±15米
        'velocity_bounds': [-5.0, 5.0],
        'orientation_bounds': [-1.0, 1.0],
        'target_bounds': [-30.0, 30.0],  # 目标可能在房间任意两点间
        'depth_bounds': [0.0, 10.0]
    }