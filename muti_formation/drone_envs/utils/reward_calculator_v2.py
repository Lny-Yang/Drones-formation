"""
🎯 极简版奖励计算模块 - 无冗余设计
"""
import numpy as np
import pybullet as p
from typing import Dict, Any, Tuple, Optional


class RewardCalculator:
    """🎯 极简版奖励计算器 - 3组件无冗余设计
    
    核心理念：
    1. 一个行为，一个奖励 - 消除功能重叠
    2. 稀疏主导方向 - 成功占比 > 90%
    3. 密集提供梯度 - 只保留必要信号
    4. 场景化奖励 - 根据环境动态调整
    
    奖励架构（4组件）：
    
    📍 稀疏奖励层 (方向引导)
    1. success: +2000 - 成功到达目标
    2. crash: -1500 - 碰撞失败
    
    📊 密集奖励层 (梯度提供)  
    3. navigation: ~1.5/step - 导航主信号
       └ 合并: 距离变化 + 朝向对齐
       └ 来源: navigation + forward_movement
    
    4. safe_navigation: ~0.5/step - 安全导航
       └ 融合: 避障 + 转向 + 速度调节
       └ 来源: obstacle + rotation + adaptive_speed
    
    奖励分布示例：
    - 快速成功(60步): +2000 +90 +30 = +2120 (成功占94.3%)
    - 慢速成功(200步): +2000 +300 +100 = +2400 (成功占83.3%)
    - 碰撞失败(150步): -1500 +150 +75 = -1275 (负值✓)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化奖励计算器
        
        Args:
            config: 奖励配置参数
        """
        # 稀疏奖励配置
        self.success_bonus = config.get('success_bonus', 2000.0)
        self.crash_penalty = config.get('crash_penalty', -1500.0)
        
        # 其他参数
        self.collision_distance = config.get('collision_distance', 0.6)
        
        # 状态记录
        self.previous_distances = {}  # 记录上一步距离
        
    def compute_total_reward(self,
                           drone_id: str,
                           position: np.ndarray,
                           target_position: np.ndarray,
                           velocity: np.ndarray,
                           depth_info: Dict[str, float],
                           orientation: Optional[np.ndarray] = None,
                           formation_info: Optional[Dict[str, Any]] = None,
                           done: bool = False,
                           success: bool = False,
                           current_step: int = 0) -> Tuple[float, Dict[str, float]]:
        """
        🎯 极简版奖励计算 - 4组件无冗余设计
        
        奖励组成：
        1. success (稀疏): +2000 - 成功占94%
        2. crash (稀疏): -1500 - 确保碰撞负值
        3. navigation (密集): ~1.5/step - 距离+朝向
        4. safe_navigation (密集): ~0.5/step - 避障+转向+速度
        
        预期总奖励：
        - 快速成功(60步): +2120 (成功占94.3%)
        - 慢速成功(200步): +2400 (成功占83.3%)
        - 碰撞失败(150步): -1275 (负值✓)
        """
        reward_details = {}

        # 1. 成功奖励 - 稀疏，最高优先级
        reward_details['success'] = self.success_bonus if success else 0.0

        # 2. 碰撞惩罚 - 稀疏，强负反馈
        reward_details['crash'] = self.crash_penalty if (done and not success) else 0.0

        # 3. 导航奖励 - 密集，合并版（距离+朝向）
        navigation_reward = self._compute_navigation_reward_merged(
            drone_id, position, target_position, velocity, orientation
        )
        reward_details['navigation'] = navigation_reward

        # 4. 安全导航奖励 - 密集，融合版（避障+转向+速度）
        safe_nav_reward = self._compute_safe_navigation_reward(
            depth_info, velocity, orientation, 
            np.linalg.norm(position - target_position)
        )
        reward_details['safe_navigation'] = safe_nav_reward

        # 计算总奖励
        total_reward = sum(reward_details.values())

        return total_reward, reward_details
    
    def _compute_navigation_reward_merged(self, drone_id: str, position: np.ndarray, 
                                         target_position: np.ndarray,
                                         velocity: np.ndarray,
                                         orientation: Optional[np.ndarray]) -> float:
        """🎯 合并版导航奖励 - 消除冗余
        
        合并功能：
        1. 距离变化奖励（来自旧navigation）
        2. 朝向对齐奖励（来自旧forward_movement）
        
        设计原理：
        - Part A: 距离减少 = 主要信号（引导靠近）
        - Part B: 朝向对齐 = 辅助信号（防止侧滑、后退）
        
        预期输出：
        - 正常飞行：+1.5/step
        - 后退/侧滑：-0.5/step
        """
        current_distance = np.linalg.norm(position - target_position)
        
        # 初始化距离记录
        if drone_id not in self.previous_distances:
            self.previous_distances[drone_id] = current_distance
            # 第一步只给基础倒数奖励
            return 1.5 * max(0, (1.0 - current_distance / 40.0))
        
        prev_distance = self.previous_distances[drone_id]
        distance_change = prev_distance - current_distance  # 正=靠近，负=远离
        
        # 更新距离记录
        self.previous_distances[drone_id] = current_distance
        
        # ===== Part A: 距离变化奖励（主要信号）=====
        # 每0.1米靠近 = +0.8分
        if distance_change > 0.01:  # 靠近目标
            reward_distance = distance_change * 8.0
            reward_distance = min(reward_distance, 2.0)  # 单步最多+2
        elif distance_change < -0.01:  # 远离目标
            reward_distance = distance_change * 8.0
            reward_distance = max(reward_distance, -1.0)  # 单步最多-1
        else:
            reward_distance = 0.0
        
        # ===== Part B: 朝向对齐奖励（辅助信号）=====
        reward_alignment = 0.0
        
        if orientation is not None and np.linalg.norm(velocity[:2]) > 0.1:
            # 计算朝向向量
            euler = p.getEulerFromQuaternion(orientation)
            yaw = euler[2]
            heading = np.array([np.cos(yaw), np.sin(yaw)])
            
            # 计算到目标的方向
            to_target = target_position[:2] - position[:2]
            distance_2d = np.linalg.norm(to_target)
            
            if distance_2d > 0.1:
                to_target_normalized = to_target / distance_2d
                
                # 朝向与目标方向的对齐度（-1到1）
                alignment = np.dot(heading, to_target_normalized)
                
                if alignment > 0.7:  # 朝向目标（cos(45°)≈0.7）
                    reward_alignment = 0.5 * (alignment - 0.7) / 0.3  # 0到+0.5
                elif alignment < 0:  # 背对目标
                    reward_alignment = -0.3 * abs(alignment)  # 0到-0.3
                # 侧向不给奖励也不惩罚（允许绕路避障）
        
        # 合并奖励
        total_reward = reward_distance + reward_alignment
        
        return total_reward
    
    def _compute_safe_navigation_reward(self, depth_info: Dict[str, float],
                                       velocity: np.ndarray,
                                       orientation: Optional[np.ndarray],
                                       distance_to_target: float) -> float:
        """🎯 融合版安全导航奖励 - 场景化设计
        
        融合功能：
        1. 避障判断（来自旧obstacle）
        2. 转向引导（来自旧rotation_guidance）
        3. 速度调节（来自旧adaptive_speed）
        
        核心思想：
        根据前方深度场景，给出"应该怎么做"的建议：
        - 开阔: 鼓励高速直行
        - 狭窄: 鼓励转向开阔方向
        - 危险: 鼓励减速或避障
        
        预期输出：
        - 开阔高速：+1.0/step
        - 正常飞行：+0.5/step  
        - 正确避障：+0.3/step
        - 错误行为：-0.5/step
        """
        depth_map = depth_info.get('depth_map', None)
        
        if depth_map is None:
            return 0.0
        
        # ===== Step 1: 分析深度信息 =====
        h, w = depth_map.shape
        
        # 前方中央区域
        center = depth_map[h//3:2*h//3, w//3:2*w//3]
        center_valid = center[center > 0.1]
        center_depth = float(center_valid.mean()) if len(center_valid) > 0 else 0.5
        
        # 左侧区域
        left = depth_map[h//4:3*h//4, :w//3]
        left_valid = left[left > 0.1]
        left_depth = float(left_valid.mean()) if len(left_valid) > 0 else 0.5
        
        # 右侧区域
        right = depth_map[h//4:3*h//4, 2*w//3:]
        right_valid = right[right > 0.1]
        right_depth = float(right_valid.mean()) if len(right_valid) > 0 else 0.5
        
        # 计算速度和角速度
        speed_2d = np.linalg.norm(velocity[:2])
        angular_vel = depth_info.get('angular_velocity', 0.0)
        
        # ===== Step 2: 场景判断与奖励 =====
        
        # 场景A: 非常开阔（前方>6m）→ 应该高速直行
        if center_depth > 1.5:
            if speed_2d > 3.0:
                return +1.0  # 优秀！高速通过
            elif speed_2d > 2.0:
                return +0.7  # 不错
            elif speed_2d > 1.0:
                return +0.3  # 还行
            else:
                return -0.2  # 太慢了
        
        # 场景B: 较开阔（前方3-6m）→ 应该中速前进
        elif center_depth > 0.75:
            if speed_2d > 1.5:
                return +0.5  # 好
            elif speed_2d > 0.8:
                return +0.3  # 还行
            else:
                return 0.0  # 一般
        
        # 场景C: 接近障碍（前方2-3m）→ 应该转向或减速
        elif center_depth > 0.5:
            # 检查应该转向哪边
            openness_diff = abs(left_depth - right_depth)
            
            if openness_diff > 0.3:  # 有明显的开阔方向
                should_turn_left = left_depth > right_depth
                is_turning_correctly = (should_turn_left and angular_vel < -0.05) or \
                                      (not should_turn_left and angular_vel > 0.05)
                
                if is_turning_correctly:
                    return +0.4  # 好！正在转向开阔方向
                elif abs(angular_vel) > 0.05:
                    return -0.2  # 转错方向了
                elif speed_2d < 1.0:
                    return +0.2  # 至少在减速
                else:
                    return -0.3  # 应该转向或减速
            else:
                # 两边差不多，减速即可
                if speed_2d < 1.0:
                    return +0.3
                else:
                    return -0.2
        
        # 场景D: 非常危险（前方<2m）→ 应该紧急避障
        else:
            if abs(angular_vel) > 0.1:  # 在旋转避障
                return +0.3
            elif speed_2d < 0.5:  # 在减速
                return +0.2
            else:
                return -0.8  # 危险！应该避障
        
        return 0.0
    
    def reset_state(self):
        """重置状态（用于新回合）"""
        self.previous_distances.clear()


def create_default_reward_config() -> Dict[str, Any]:
    """🎯 极简版奖励配置
    
    核心设计：
    1. 稀疏奖励主导（成功占比>90%）
    2. 密集奖励精简（无冗余）
    3. 碰撞必为负值
    
    奖励测算：
    - 快速成功(60步): +2000 +90 +30 = +2120 (成功占94.3%)
    - 慢速成功(200步): +2000 +300 +100 = +2400 (成功占83.3%)
    - 碰撞失败(150步): -1500 +150 +75 = -1275 (负值✓)
    """
    return {
        # 稀疏奖励
        'success_bonus': 2000.0,         # 成功奖励（提高到2000）
        'crash_penalty': -1500.0,        # 碰撞惩罚（提高到-1500）
        
        # 避障参数
        'collision_distance': 0.6,       # 碰撞阈值：0.6米
        
        # 深度处理器参数
        'depth_scale': 4.0,              # 深度缩放因子
        'max_depth': 2.0,                # 最大深度值
        'cnn_feature_dim': 128,          # CNN特征维度
    }
