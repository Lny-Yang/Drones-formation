"""
多无人机编队的奖励计算模块
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .depth_obstacle_processor import DepthObstacleProcessor

class RewardCalculator:
    """🎯 极简版奖励计算器 - 3组件无冗余设计
    
    核心理念：
    1. 一个行为，一个奖励 - 消除功能重叠
    2. 稀疏主导方向 - 成功占比 > 90%
    3. 密集提供梯度 - 只保留必要信号
    4. 场景化奖励 - 根据环境动态调整
    
    奖励架构（3组件）：
    
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
        # 深度处理器配置
        collision_distance = config.get('collision_distance', 0.6)
        self.depth_processor_config = {
            'collision_threshold': collision_distance / config.get('depth_scale', 4.0),
            'depth_scale': config.get('depth_scale', 4.0),
            'max_depth': config.get('max_depth', 2.0),
            'cnn_feature_dim': config.get('cnn_feature_dim', 128)
        }
        
        # 🎯 极简奖励系统：确保稀疏主导
        self.success_bonus = config.get('success_bonus', 2000.0)   # 成功奖励（提高到2000）
        self.crash_penalty = config.get('crash_penalty', -1500.0)  # 碰撞惩罚（提高到-1500）
        
        # 其他参数
        self.collision_distance = config.get('collision_distance', 0.6)
        
        # 用于状态记录
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
        🎯 极简版奖励计算 - 3组件无冗余设计
        
        核心改进：
        1. 消除功能重叠 - navigation合并了forward_movement
        2. 融合相关功能 - safe_navigation融合了obstacle+rotation+speed
        3. 提高稀疏占比 - 成功从85%提升到94%
        4. 确保负值惩罚 - 碰撞失败必为负值
        
        奖励组成（4个独立组件）：
        
        1. **success** (稀疏): +2000
           - 最高优先级，明确目标
           - 占比提升到94%（快速成功）
        
        2. **crash** (稀疏): -1500
           - 强负反馈，确保碰撞必为负值
           - 计算: -1500 + 300(密集最多) = -1200 ✓
        
        3. **navigation** (密集): ~1.5/step
           - 合并: 距离变化 + 朝向对齐
           - 来源: 旧navigation + 旧forward_movement
           - 功能: 引导靠近目标，防止侧滑
        
        4. **safe_navigation** (密集): ~0.5/step
           - 融合: 避障 + 转向 + 速度调节
           - 来源: 旧obstacle + 旧rotation + 旧adaptive_speed
           - 功能: 根据深度场景给出安全导航建议
        
        预期总奖励：
        - 快速成功(60步): +2000 +90 +30 = +2120 (成功占94.3%)
        - 慢速成功(200步): +2000 +300 +100 = +2400 (成功占83.3%)
        - 碰撞失败(150步): -1500 +150 +75 = -1275 (负值✓)
        """
        reward_details = {}

        # 1. 成功奖励 - 最高优先级
        if success:
            reward_details['success'] = self.success_bonus
        else:
            reward_details['success'] = 0.0

        # 2. 碰撞惩罚 - 强负反馈
        if done and not success:
            reward_details['crash'] = self.crash_penalty
        else:
            reward_details['crash'] = 0.0

        # 3. 导航奖励 - 合并版（距离+朝向）
        navigation_reward = self._compute_navigation_reward_merged(
            drone_id, position, target_position, velocity, orientation
        )
        reward_details['navigation'] = navigation_reward

        # 4. 安全导航奖励 - 融合版（避障+转向+速度）
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
        
        Args:
            drone_id: 无人机ID
            position: 当前位置
            target_position: 目标位置
            velocity: 速度向量
            orientation: 朝向四元数
            
        Returns:
            合并后的导航奖励值（-1到+3范围）
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
            import pybullet as p
            
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
        
        Args:
            depth_info: 深度信息字典
            velocity: 速度向量
            orientation: 朝向四元数
            distance_to_target: 到目标的距离
            
        Returns:
            融合后的安全导航奖励（-1到+1范围）
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
        distance_inverse_reward = 2.0 * max(0, (1.0 - current_distance / 40.0))
        
        # 合并奖励
        total_reward = change_reward + distance_inverse_reward
        
        return total_reward
    
    def _compute_minimal_obstacle_reward(self, depth_info: Dict[str, float], 
                                        distance_to_target: float) -> float:
        """🔧 极简版避障奖励 - 只处理危险情况
        
        设计原则：
        1. 只在真正危险时才给信号
        2. 权重降低80%，避免过度谨慎
        3. 让导航奖励主导，避障只是安全约束
        
        目标：
        - 正常飞行：每步+0.3（开阔空间）
        - 接近障碍：每步-0.5（危险警告）
        - 碰撞前：由环境crash_penalty处理
        
        Args:
            depth_info: 深度信息字典
            distance_to_target: 到目标的距离
            
        Returns:
            避障奖励值（-1到+1范围）
        """
        depth_map = depth_info.get('depth_map', None)
        
        if depth_map is None:
            return 0.0
        
        # 计算前方最小深度
        h, w = depth_map.shape
        center = depth_map[h//3:2*h//3, w//3:2*w//3]
        valid = center[center > 0.1]
        
        if len(valid) == 0:
            return 0.0
        
        min_depth = float(np.min(valid))
        
        # 简化的避障奖励
        if min_depth < 0.5:  # 非常危险（<2m）
            return -0.5
        elif min_depth < 0.75:  # 警戒区域（<3m）
            return -0.2
        elif min_depth > 1.5:  # 非常开阔（>6m）
            return 0.5
        elif min_depth > 1.0:  # 较开阔（>4m）
            return 0.3
        else:
            return 0.0
    
    def _compute_navigation_reward(self, drone_id: str, position: np.ndarray, target_position: np.ndarray) -> float:
        """🔧 精简版：导航奖励 - 合并distance和progress，消除冗余
        
        设计原理：
        1. 合并了distance和progress两个高度重合的组件（90%重合）
        2. 保留两者优势：距离变化激励 + 距离倒数奖励
        3. 使用更高的multiplier（70）补偿删除distance
        
        公式：
        - 距离减少: (prev_dist - curr_dist) × 70
        - 距离倒数奖励: 10 × (1 - curr_dist/max_dist)
        
        预期效果：
        - 60步成功: 每步平均0.15m × 70 × 60 = 630 + 600(倒数) = 1230分
        - 比旧版(progress 900 + distance 420 = 1320)稍低，但更清晰
        
        Args:
            drone_id: 无人机ID
            position: 当前位置
            target_position: 目标位置
            
        Returns:
            导航奖励值
        """
        current_distance = np.linalg.norm(position - target_position)
        
        # 初始化或获取上一步的距离
        if drone_id not in self.previous_distances:
            self.previous_distances[drone_id] = current_distance
            # 第一步给基础奖励
            base_reward = 10.0 * max(0, (1.0 - current_distance / 40.0))
            return base_reward
        
        prev_distance = self.previous_distances[drone_id]
        distance_change = prev_distance - current_distance  # 正值=靠近，负值=远离
        
        # 更新距离记录
        self.previous_distances[drone_id] = current_distance
        
        # 🔧 合并后的导航奖励计算
        navigation_multiplier = 20.0  # 🔧 从70.0大幅降低到20.0，控制每步奖励
        
        # 组件1: 距离变化奖励（主要信号）
        if distance_change > 0.002:  # 靠近目标
            change_reward = distance_change * navigation_multiplier
            # 🔧 修复：简化加成计算，避免爆炸
            distance_factor = 1.0 - min(current_distance / 40.0, 1.0)  # 0到1之间
            change_reward = change_reward + distance_factor * 1.0  # 🔧 从3降到1
            change_reward = min(change_reward, 4.0)  # 🔧 从15降到4，单步最多+4
        elif distance_change < -0.002:  # 远离目标
            # 轻微惩罚，允许绕路避障
            change_reward = distance_change * navigation_multiplier * 0.2
            change_reward = max(change_reward, -1.0)  # 🔧 从-3降到-1，最多-1/步
        else:
            # 距离基本不变
            change_reward = 0.0
        
        # 组件2: 距离倒数奖励（辅助信号，鼓励接近目标）
        distance_inverse_reward = 3.0 * max(0, (1.0 - current_distance / 40.0))  # 🔧 从10降到3
        
        # 合并
        total_navigation_reward = change_reward + distance_inverse_reward
        
        return total_navigation_reward
    
    
    # ==================================================================================
    # 🗑️ 以下函数已废弃，保留仅为向后兼容，实际不再使用
    # ==================================================================================
    
    def _compute_distance_reward(self, drone_id: str, position: np.ndarray, target_position: np.ndarray) -> float:
        """❌ 已废弃：被_compute_navigation_reward替代（与progress 90%重合）"""
        return 0.0
    
    def _compute_progress_reward(self, drone_id: str, position: np.ndarray, target_position: np.ndarray) -> float:
        """❌ 已废弃：被_compute_navigation_reward替代（与distance合并）"""
        return 0.0
    
    def _compute_exploration_reward(self, drone_id: str, position: np.ndarray, depth_info: Dict[str, float]) -> float:
        """❌ 已废弃：固定翼模式不需要探索"""
        return 0.0
    
    def _compute_fixed_wing_speed_reward(self, velocity: np.ndarray, depth_info: Dict[str, float]) -> float:
        """❌ 已废弃：被_compute_adaptive_speed_reward替代（合并step_penalty）"""
        return 0.0
    
    # ==================================================================================
    
    def _compute_balanced_obstacle_reward(self, depth_info: Dict[str, float], distance_to_target: float) -> float:
        """🔧 重构：基于障碍物分析计算避障奖励

        职责：根据DepthObstacleProcessor提供的障碍物信息计算奖励
        不再依赖深度处理器的奖励计算，完全自主决策

        奖励策略：
        1. 即时碰撞：强惩罚（由环境done处理，这里返回0）
        2. 危险接近（<1.5m）：中等惩罚，鼓励保持安全距离
        3. 安全距离（1.5-2.5m）：小奖励，鼓励安全通过
        4. 远离障碍（>2.5m）：不奖励，让导航主导

        动态权重调整：
        - 远离目标时：降低避障权重，鼓励快速前进
        - 接近目标时：提高避障权重，确保安全到达

        Args:
            depth_info: 包含障碍物分析信息的字典
            distance_to_target: 到目标的距离

        Returns:
            避障奖励
        """
        depth_map = depth_info.get('depth_map', None)

        if depth_map is None:
            return 0.0

        # 🔧 使用新的障碍物分析接口
        processor = DepthObstacleProcessor(**self.depth_processor_config)
        obstacle_info = processor.get_obstacle_analysis(depth_map)

        # 提取关键信息
        physical_min_depth = obstacle_info['physical_min_depth']
        danger_level = obstacle_info['danger_level']
        forward_openness = obstacle_info['forward_openness']
        is_imminent = obstacle_info['is_imminent_collision']

        # 计算基础避障奖励
        base_obstacle_reward = 0.0

        if is_imminent:
            # 即时碰撞危险：让环境的crash_penalty处理，这里不重复惩罚
            base_obstacle_reward = 0.0
        elif physical_min_depth < 1.0:
            # 危险接近（<1.0m）：轻微惩罚
            base_obstacle_reward = -0.2 * danger_level  # 🔧 从-0.5降到-0.2
        elif physical_min_depth < 1.5:
            # 警戒区域（1.0-1.5m）：几乎不惩罚
            base_obstacle_reward = -0.1 * danger_level  # 🔧 从-0.2降到-0.1
        elif physical_min_depth < 2.5:
            # 安全通过区域（1.5-2.5m）：小奖励
            base_obstacle_reward = 0.2 * (1.0 - danger_level)  # 🔧 从0.5降到0.2
        else:
            # 远离障碍（>2.5m）：不奖励，让导航主导
            base_obstacle_reward = 0.0

        # 🔥 前方开放奖励：鼓励朝向开阔空间
        openness_reward = forward_openness * 0.5  # 🔧 从2.0降到0.5

        # 组合奖励
        total_obstacle_reward = base_obstacle_reward + openness_reward

        # 🔧 动态权重调整：根据到目标的距离（整体降低）
        if distance_to_target > 8.0:
            # 远距离：降低避障权重
            adjusted_reward = total_obstacle_reward * 0.5  # 🔧 从0.8降到0.5
        elif distance_to_target > 3.0:
            # 中距离：保持避障权重
            adjusted_reward = total_obstacle_reward * 0.7  # 🔧 从1.0降到0.7
        else:
            # 近距离：提高避障权重
            adjusted_reward = total_obstacle_reward * 1.0  # 🔧 从1.2降到1.0

        return adjusted_reward
    
    def _compute_forward_movement_reward(self, position: np.ndarray, target_position: np.ndarray,
                                        velocity: np.ndarray, orientation: Optional[np.ndarray],
                                        depth_info: Optional[Dict[str, Any]] = None) -> float:
        """🔥 新增：前进行为奖励 - 明确鼓励朝目标方向移动 + 防止侧滑
        
        问题1：网络学到"面向障碍物但后退"的策略，因为后退更安全
        问题2：无人机侧滑（朝向≠移动方向），摄像头看不到碰撞方向
        
        解决：
        1. 奖励"朝向目标方向的速度分量"，惩罚后退行为
        2. 惩罚"朝向与速度不一致"（侧滑），确保移动方向=摄像头方向
        
        Args:
            position: 当前位置
            target_position: 目标位置  
            velocity: 速度向量 [vx, vy, vz]
            orientation: 朝向四元数（可选）
            
        Returns:
            前进行为奖励
        """
        # 计算目标方向向量
        to_target = target_position - position
        distance = np.linalg.norm(to_target)
        
        if distance < 0.01:  # 已到达目标
            return 0.0
            
        # 归一化目标方向
        to_target_normalized = to_target / distance
        
        # 计算速度在目标方向上的投影（标量投影）
        velocity_2d = velocity[:2]  # 只考虑平面速度
        forward_velocity = np.dot(velocity_2d, to_target_normalized[:2])
        speed_2d = np.linalg.norm(velocity_2d)
        
        # 🔥 关键优化：根据前方深度动态调整速度要求
        # 避免在狭窄空间过度惩罚慢速（慢速是必要的）
        depth_map = depth_info.get('depth_map', None) if depth_info else None
        if depth_map is not None:
            h, w = depth_map.shape
            center = depth_map[h//3:2*h//3, w//3:2*w//3]
            valid_center = center[center > 0.1]
            center_depth = valid_center.mean() if len(valid_center) > 0 else 0.5
        else:
            center_depth = 1.0  # 默认中等开阔度
        
        # 根据前方开阔度设置速度要求和奖励
        speed_bonus = 0.0
        if center_depth > 1.5:  # 非常开阔 (>6m) - 要求高速
            if speed_2d < 1.5:
                speed_bonus = -0.5  # 🔧 大幅降低惩罚，从-2降到-0.5
            elif speed_2d < 3.0:
                speed_bonus = -0.2  # 🔧 从-1降到-0.2
            elif speed_2d >= 5.0:
                speed_bonus = +1.0  # 🔧 从5降到1
            elif speed_2d >= 3.0:
                speed_bonus = +0.5  # 🔧 从2降到0.5
        elif center_depth > 0.75:  # 较开阔 (3-6m) - 要求中速
            if speed_2d < 0.8:
                speed_bonus = -0.5  # 🔧 从-2降到-0.5
            elif speed_2d < 1.5:
                speed_bonus = -0.2  # 🔧 从-1降到-0.2
            elif speed_2d >= 3.0:
                speed_bonus = +0.8  # 🔧 从3降到0.8
            elif speed_2d >= 1.5:
                speed_bonus = +0.4  # 🔧 从1.5降到0.4
        else:  # 狭窄 (<3m) - 允许慢速，只要在动
            if speed_2d < 0.3:
                speed_bonus = -0.3  # 🔧 从-1.5降到-0.3
            elif speed_2d >= 1.0:
                speed_bonus = +0.3  # 🔧 从1.5降到0.3
        
        # 组件1: 前进/后退奖励
        forward_reward = 0.0
        if forward_velocity > 0.01:  # 向目标前进
            # 🔧 修复：大幅降低上限，从10.0降到2.0
            # 目标：单步约1-2分，60步约60-120分
            forward_reward = min(forward_velocity * 0.8, 2.0)  # 🔧 从10.0降到2.0
        elif forward_velocity < -0.01:  # 后退（远离目标）
            # 惩罚后退行为
            forward_reward = forward_velocity * 2.0  # 🔧 从5降到2
            forward_reward = max(forward_reward, -2.0)  # 🔧 从-5降到-2
        
        # 组件2: 朝向对齐奖励（防止侧滑）
        alignment_reward = 0.0
        if orientation is not None:
            import pybullet as p
            # 从四元数获取yaw角度
            euler = p.getEulerFromQuaternion(orientation)
            yaw = euler[2]
            
            # 计算机头方向向量
            heading_x = np.cos(yaw)
            heading_y = np.sin(yaw)
            heading = np.array([heading_x, heading_y])
            
            # 计算速度方向（归一化）
            speed = np.linalg.norm(velocity_2d)
            if speed > 0.05:  # 只在有明显速度时检查对齐
                velocity_direction = velocity_2d / speed
                
                # 计算朝向与速度方向的点积（-1到1）
                # 1.0 = 完全对齐（正向），-1.0 = 完全相反（倒退），0 = 侧滑
                alignment = np.dot(heading, velocity_direction)
                
                # 🔥 关键：惩罚侧滑（alignment接近0）
                if alignment < 0.7:  # cos(45°) ≈ 0.7
                    # 侧滑惩罚：速度越大、偏离越多，惩罚越重
                    # 例如：speed=1.0, alignment=0 → -0.5惩罚
                    alignment_reward = -0.5 * speed * (1.0 - abs(alignment))
                    alignment_reward = max(alignment_reward, -1.0)  # 限制最大惩罚
        
        return forward_reward + alignment_reward + speed_bonus
    
    def _compute_rotation_guidance_reward(self, depth_info: Dict[str, float], 
                                         orientation: Optional[np.ndarray],
                                         velocity: np.ndarray) -> float:
        """🔥 优化：旋转导航奖励 - 奖励正确方向的实际旋转行为
        
        问题：无人机面对墙壁时不知道该往哪转，或者知道但不执行
        解决：检测左右哪边更开阔，奖励朝那个方向的实际旋转动作
        
        Args:
            depth_info: 深度信息字典，包含depth_map和angular_velocity
            orientation: 当前朝向四元数
            velocity: 速度向量
            
        Returns:
            旋转导航奖励
        """
        depth_map = depth_info.get('depth_map', None)
        if depth_map is None or orientation is None:
            return 0.0
        
        h, w = depth_map.shape
        
        # 分析三个方向的开阔度
        # 前方中央区域
        center = depth_map[h//3:2*h//3, w//3:2*w//3]
        # 左侧区域
        left = depth_map[h//4:3*h//4, :w//3]
        # 右侧区域
        right = depth_map[h//4:3*h//4, 2*w//3:]
        
        # 计算每个区域的平均深度
        def avg_depth(region):
            valid = region[region > 0.1]
            return valid.mean() if len(valid) > 0 else 0.0
        
        center_depth = avg_depth(center)
        left_depth = avg_depth(left)
        right_depth = avg_depth(right)
        
        # 获取实际角速度（rad/s）和速度
        angular_vel = depth_info.get('angular_velocity', 0.0)
        speed_2d = np.linalg.norm(velocity[:2]) if len(velocity) >= 2 else 0.0
        
        # 🔥 重要：深度图已归一化到 [0, 2.0]范围
        # depth_scale = 4.0, max_depth = 2.0
        # 归一化值 0.5 = 实际 2.0m
        # 归一化值 1.0 = 实际 4.0m
        # 归一化值 1.5 = 实际 6.0m
        # 归一化值 2.0 = 实际 8.0m
        
        # 🔥 策略优先级：前方开阔 → 专注前进，不要旋转！
        # rotation_guidance 只负责旋转方向，不负责速度（速度由forward_movement管）
        
        # 情况1：前方非常开阔（>6m），完全不应该旋转
        if center_depth > 1.5:  # 归一化值1.5 = 实际6m
            return 0.0  # 不给旋转奖励/惩罚，让forward_movement主导
        
        # 情况2：前方较开阔（>4m）且已经在快速前进，继续前进
        if center_depth > 1.0 and speed_2d > 2.0:  # 归一化值1.0 = 实际4m
            return 0.0  # 不干扰
        
        # 情况3：前方较开阔（>3m）且在正常前进，鼓励继续
        if center_depth > 0.75 and speed_2d > 1.5:  # 归一化值0.75 = 实际3m
            return 0.0
        
        # 前方被堵（center < 1.5，即<6m），分析侧面情况
        max_side_depth = max(left_depth, right_depth)
        
        # 情况1：两侧都很近，被困住了，鼓励任意方向旋转
        if max_side_depth < 0.25:  # 归一化值0.25 = 实际1.0m，真的被困
            rotation_speed = abs(angular_vel)
            if rotation_speed > 0.01:  # 只要在旋转就给奖励
                return min(rotation_speed * 10.0, 5.0)  # 最高+5.0
            else:
                # 被困但不旋转，给负奖励
                return -1.0
        
        # 情况2：确定应该转向哪边（左或右）
        openness_diff = abs(left_depth - right_depth)
        
        # 🔥 关键修复3：使用归一化值！0.3 = 实际1.2m差异
        if openness_diff < 0.3:  # 归一化值0.3 = 实际1.2m差异
            return 0.0  # 两边差不多，不给奖励
        
        # 确定目标旋转方向
        if left_depth > right_depth:
            target_direction = -1  # 应该左转（逆时针，负角速度）
        else:
            target_direction = 1  # 应该右转（顺时针，正角速度）
        
        # 🔥 核心：检查是否在朝正确方向旋转
        rotation_speed = abs(angular_vel)
        
        # 判断旋转方向是否正确
        is_rotating_correctly = (angular_vel * target_direction) > 0
        
        if is_rotating_correctly:
            # 朝正确方向旋转，奖励 = 开阔度差异 × 旋转速度
            reward = min(openness_diff * rotation_speed * 0.5, 0.5)  # 🔧 从1.5降到0.5
            return reward
        elif rotation_speed > 0.01:
            # 在旋转但方向错误，轻微惩罚
            return -0.3  # 🔧 从-1.5降到-0.3
        else:
            # 应该旋转但没有旋转，惩罚
            return -0.2  # 🔧 从-1.0降到-0.2
    
    def _compute_adaptive_speed_reward(self, velocity: np.ndarray, depth_info: Dict[str, float], 
                                      distance_to_target: float) -> float:
        """🚀 精简版：自适应速度奖励 - 合并fixed_wing_speed和step_penalty
        
        设计原理：
        1. 合并fixed_wing_speed（速度约束）和step_penalty（效率激励）
        2. 根据环境开阔度和到目标距离动态调整速度要求
        3. 整合效率激励：快速到达更高奖励
        
        速度策略（动态）：
        - 开阔 + 远离目标: 鼓励高速（3-5 m/s）→ +2.0
        - 开阔 + 接近目标: 鼓励中速（1.5-2.5 m/s）→ +1.5
        - 狭窄: 允许低速（0.5-1.5 m/s）→ +0.5
        - 失速（<0.3 m/s）: 强惩罚 → -2.0
        - 效率激励: 步数越少（速度越快），额外奖励
        
        Args:
            velocity: 速度向量 [vx, vy, vz]
            depth_info: 深度信息，用于判断环境开阔度
            distance_to_target: 到目标的距离
            
        Returns:
            自适应速度奖励值
        """
        # 计算水平速度（忽略z轴）
        horizontal_speed = np.linalg.norm(velocity[:2])
        
        # 获取前方深度信息（判断是否开阔）
        depth_map = depth_info.get('depth_map', None)
        
        # 分析前方开阔度
        if depth_map is not None:
            h, w = depth_map.shape
            center = depth_map[h//3:2*h//3, w//3:2*w//3]
            valid_center = center[center > 0.1]
            center_depth = valid_center.mean() if len(valid_center) > 0 else 0.5
            # 转换为实际距离（depth_scale = 4.0）
            actual_clearance = center_depth * 4.0  # 归一化值 → 实际米数
        else:
            actual_clearance = 4.0  # 默认中等开阔
        
        # 🔧 根据环境和距离动态设置速度要求
        if distance_to_target > 8.0 and actual_clearance > 6.0:
            # 场景1: 远离目标 + 开阔 → 鼓励高速冲刺
            if horizontal_speed < 0.3:
                return -0.5  # 🔧 从-2.5降到-0.5
            elif horizontal_speed < 1.5:
                return -0.2  # 🔧 从-1.0降到-0.2
            elif 3.0 <= horizontal_speed <= 5.0:
                # 🎯 高速冲刺奖励 + 效率奖励
                return 0.5  # 🔧 从2.5降到0.5
            elif horizontal_speed >= 5.0:
                return 0.3  # 🔧 从1.0降到0.3
            else:
                return 0.2  # 🔧 从0.5降到0.2
                
        elif distance_to_target > 3.0 and actual_clearance > 3.0:
            # 场景2: 中等距离 + 较开阔 → 鼓励最优速度
            if horizontal_speed < 0.3:
                return -0.5  # 🔧 从-2.0降到-0.5
            elif horizontal_speed < 1.0:
                return -0.2  # 🔧 从-0.5降到-0.2
            elif 1.5 <= horizontal_speed <= 2.5:
                # 🎯 最优速度
                return 0.4  # 🔧 从1.7降到0.4
            elif horizontal_speed <= 4.0:
                return 0.3  # 🔧 从1.0降到0.3
            else:
                return -0.2  # 🔧 从-0.5降到-0.2
                
        elif distance_to_target > 1.0:
            # 场景3: 接近目标 → 允许降速，确保安全
            if horizontal_speed < 0.3:
                return -0.4  # 🔧 从-2.0降到-0.4
            elif 0.5 <= horizontal_speed <= 1.5:
                # 🎯 安全接近速度
                return 0.3  # 🔧 从1.0降到0.3
            elif horizontal_speed <= 2.5:
                return 0.2  # 🔧 从0.5降到0.2
            else:
                return -0.2  # 🔧 从-1.0降到-0.2
                
        else:
            # 场景4: 非常接近目标（<1m）→ 允许低速精确对准
            if horizontal_speed < 0.2:
                return -0.2  # 🔧 从-1.0降到-0.2
            elif horizontal_speed <= 1.0:
                return 0.2  # 🔧 从0.5降到0.2
            else:
                return -0.1  # 🔧 从-0.5降到-0.1
    
    def _compute_fixed_wing_speed_reward(self, velocity: np.ndarray, depth_info: Dict[str, float]) -> float:
        """🚀 固定翼模式专用：速度保持奖励 - 鼓励保持最小速度并根据环境调整
        
        设计原理：
        1. 固定翼无人机需要保持最小速度以维持升力（防止失速）
        2. 在开阔区域应该加速，在狭窄区域允许减速
        3. 惩罚过低速度（<0.3 m/s），奖励合适速度范围
        
        速度策略：
        - 失速危险（<0.3 m/s）: 强惩罚 -2.0
        - 低速（0.3-1.5 m/s）: 轻微惩罚 -0.5
        - 最优速度（1.5-2.5 m/s）: 奖励 +1.0
        - 高速（2.5-5.0 m/s）: 根据环境，开阔时奖励，狭窄时惩罚
        - 过速（>5.0 m/s）: 惩罚 -1.0（难以避障）
        
        Args:
            velocity: 速度向量 [vx, vy, vz]
            depth_info: 深度信息，用于判断环境开阔度
            
        Returns:
            速度奖励值
        """
        # 计算水平速度（忽略z轴）
        horizontal_speed = np.linalg.norm(velocity[:2])
        
        # 获取前方深度信息（判断是否开阔）
        depth_map = depth_info.get('depth_map', None)
        
        # 分析前方开阔度
        if depth_map is not None:
            h, w = depth_map.shape
            center = depth_map[h//3:2*h//3, w//3:2*w//3]
            valid_center = center[center > 0.1]
            center_depth = valid_center.mean() if len(valid_center) > 0 else 0.5
            # 转换为实际距离（depth_scale = 4.0）
            actual_clearance = center_depth * 4.0  # 归一化值 → 实际米数
        else:
            actual_clearance = 4.0  # 默认中等开阔
        
        # 根据速度和环境计算奖励
        speed_reward = 0.0
        
        # 配置参数（从config获取）
        min_speed = 0.3  # 最小速度阈值
        optimal_min = 1.5  # 最优速度范围下限
        optimal_max = 2.5  # 最优速度范围上限
        
        if horizontal_speed < min_speed:
            # ⚠️ 失速危险：强惩罚
            speed_reward = -0.5  # 🔧 从-2.0降到-0.5
            
        elif horizontal_speed < optimal_min:
            # 低速：轻微惩罚，鼓励加速
            speed_reward = -0.2  # 🔧 从-0.5降到-0.2
            
        elif optimal_min <= horizontal_speed <= optimal_max:
            # ✅ 最优速度范围：给予奖励
            speed_reward = 0.3  # 🔧 从1.0降到0.3
            
        elif horizontal_speed <= 5.0:
            # 高速：根据环境判断
            if actual_clearance > 6.0:
                # 开阔环境：鼓励高速
                speed_reward = 0.4  # 🔧 从1.5降到0.4
            elif actual_clearance > 3.0:
                # 中等开阔：允许但不特别鼓励
                speed_reward = 0.2  # 🔧 从0.5降到0.2
            else:
                # 狭窄环境：惩罚高速（危险）
                speed_reward = -0.3  # 🔧 从-1.0降到-0.3
                
        else:
            # 过速（>5.0 m/s）：难以避障，惩罚
            speed_reward = -0.3  # 🔧 从-1.0降到-0.3
        
        return speed_reward
    
    def reset_state(self):
        """重置状态（用于新回合）"""
        self.previous_distances.clear()


def create_default_reward_config() -> Dict[str, Any]:
    """🔧 重构版奖励配置 - 平衡尺度，简化组件
    
    核心设计：
    1. 稀疏奖励（成功/碰撞）主导方向
    2. 密集奖励（导航/避障）提供梯度
    3. 确保尺度平衡：成功 >> 碰撞 >> 密集累积
    
    奖励尺度测算：
    - 成功(60步): +1000(稀疏) +120(导航) +30(避障) -30(步数) = +1120
    - 成功(200步): +1000 +400 +100 -100 = +1400
    - 碰撞(150步): -800 +300 +75 -75 = -500 ✓
    
    成功占比：
    - 快速: 89.3% ✓
    - 慢速: 71.4% ✓
    """
    return {
        # 核心奖励 - 稀疏信号
        'success_bonus': 1000.0,         # 成功奖励（主要目标）
        'crash_penalty': -800.0,         # 碰撞惩罚（确保负值）
        
        # 步数惩罚 - 每步固定
        'step_penalty_per_step': -0.5,   # 每步-0.5，鼓励快速到达
        'max_episode_steps': 3000,       # 最大步数
        
        # 避障参数
        'collision_distance': 0.6,       # 碰撞阈值：0.6米
        
        # 深度处理器参数
        'depth_scale': 4.0,              # 深度缩放因子
        'max_depth': 2.0,                # 最大深度值
        'cnn_feature_dim': 128,          # CNN特征维度
    }