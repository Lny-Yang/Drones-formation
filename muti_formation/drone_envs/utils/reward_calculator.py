"""
🎯 极简版奖励计算模块 - 无冗余设计
"""
import numpy as np
import pybullet as p
from typing import Dict, Any, Tuple, Optional
from .depth_obstacle_processor import DepthObstacleProcessor

class RewardCalculator:
    """🎯 极简版奖励计算器 - 平衡稳定版 v5（取消密集奖励上限）
    
    核心理念：
    1. 一个行为，一个奖励 - 消除功能重叠
    2. 🔥 成功奖励占比要足够大 - 引导正确的学习方向
    3. 🔥 碰撞惩罚明确 - 确保错误行为有代价
    4. 🔥 无密集奖励上限 - 不阻碍智能体学习复杂路径
    5. 场景化奖励 - 根据环境动态调整
    
    奖励架构（4组件）：
    
    📍 稀疏奖励层 (方向引导)
    1. success: +10000 - 成功到达目标（占比大，引导学习）
    2. crash: -2000 - 碰撞失败（确保负值）
    
    📊 密集奖励层 (梯度提供) - 无上限限制
    3. navigation: ~3.0/step - 导航主信号
       └ 合并: 距离变化 + 朝向对齐
       └ 来源: navigation + forward_movement
    
    4. safe_navigation: ~1.0/step - 安全导航
       └ 融合: 避障 + 转向 + 速度调节
       └ 来源: obstacle + rotation + adaptive_speed
    
    实际数据分析：
    - 3000步超时实际密集奖励：≈5500分
    - 正常导航密集奖励：≈2000-3000分
    - 理论最大密集奖励：≈21000分（极端情况）
    
    奖励分布示例（v5 - 简化版）：
    - 快速成功(60步): +10000 +420 = +10420 (成功占96%✓)
    - 慢速成功(200步): +10000 +1400 = +11400 (成功占88%✓)
    - 碰撞失败(150步): -2000 +1000 = -1000 (负值✓，避免碰撞)
    - 超时失败(3000步): 0 +5500 = +5500 (明显低于成功✓，鼓励效率)
    
    设计理念：
    ✅ 成功占比大（88-96%），引导正确方向
    ✅ 碰撞必为负值，明确错误代价
    ✅ 无上限限制，允许智能体充分探索和学习
    ✅ 简单直接，易于调试和理解
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化奖励计算器
        
        Args:
            config: 奖励配置参数
        """
        # 稀疏奖励配置
        self.success_bonus = config.get('success_bonus', 10000.0)
        self.crash_penalty = config.get('crash_penalty', -2000.0)
        
        # 其他参数
        self.collision_distance = config.get('collision_distance', 0.6)
        
        # 状态记录
        self.previous_distances = {}  # 记录上一步距离
        
        # ✅ 初始化深度处理器（用于专业的障碍物分析）
        self.depth_processor = DepthObstacleProcessor(
            depth_image_size=(128, 160),
            collision_threshold=config.get('collision_distance', 0.6),
            depth_scale=config.get('depth_scale', 4.0),
            max_depth=config.get('max_depth', 2.0),
            cnn_feature_dim=config.get('cnn_feature_dim', 128)
        )

        # 密集奖励缩放系数（建议0.2~0.3）
        self.dense_scale = config.get('dense_scale', 0.2)
        
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
        🎯 极简版奖励计算 - 平衡稳定版 v5（取消密集奖励上限）
        
        奖励组成：
        1. success (稀疏): +10000 - 成功占比大，引导学习
        2. crash (稀疏): -2000 - 碰撞惩罚，确保负值
        3. navigation (密集): ~3.0/step - 距离+朝向（无上限）
        4. safe_navigation (密集): ~1.0/step - 避障+转向+速度（无上限）
        
        预期总奖励（v5 - 简化版）：
        - 快速成功(60步): +10000 +420 = +10420 (成功占96%✓)
        - 慢速成功(200步): +10000 +1400 = +11400 (成功占88%✓)
        - 碰撞失败(150步): -2000 +1000 = -1000 (负值✓)
        - 超时失败(3000步): 0 +5500 = +5500 (明显低于成功✓)
        
        设计理念：
        ✅ 成功占比大（88-96%），引导正确方向
        ✅ 碰撞必为负值，明确错误代价
        ✅ 无上限限制，允许智能体充分探索和学习
        ✅ 简单直接，易于调试
        """
        reward_details = {}

        # 1. 成功奖励 - 稀疏，最高优先级
        reward_details['success'] = self.success_bonus if success else 0.0

        # 2. 碰撞惩罚 - 稀疏，强负反馈
        collision_occurred = depth_info.get('collision', False)
        reward_details['crash'] = self.crash_penalty if collision_occurred else 0.0
        
        # 🔥 3. 超时惩罚 - 稀疏，防止"拖时间"策略
        # 如果回合结束但既没成功也没碰撞，说明是超时
        # -8000确保：成功(+5900) >> 碰撞(-950) > 超时(-1000)（max_steps=1000时）
        # 超时总奖励 = -8000 + 7000(密集) = -1000（强负值，必须避免！）
        timeout_occurred = done and not success and not collision_occurred
        reward_details['timeout'] = -8000.0 if timeout_occurred else 0.0

        # 4. 导航奖励 - 密集，合并版（距离+朝向）
        navigation_reward = self._compute_navigation_reward_merged(
            drone_id, position, target_position, velocity, orientation
        )
        # 5. 安全导航奖励 - 密集，融合版（避障+转向+速度）
        safe_nav_reward = self._compute_safe_navigation_reward(
            depth_info, velocity, orientation, 
            np.linalg.norm(position - target_position)
        )
        # 统一缩放密集奖励
        reward_details['navigation'] = navigation_reward * self.dense_scale
        reward_details['safe_navigation'] = safe_nav_reward * self.dense_scale

        # 计算总奖励
        total_reward = sum(reward_details.values())

        return total_reward, reward_details
    
    def _compute_navigation_reward_merged(self, drone_id: str, position: np.ndarray, 
                                         target_position: np.ndarray,
                                         velocity: np.ndarray,
                                         orientation: Optional[np.ndarray]) -> float:
        """🎯 合并版导航奖励 - 消除冗余（v2: 2倍增强）
        
        合并功能：
        1. 距离变化奖励（来自旧navigation）
        2. 朝向对齐奖励（来自旧forward_movement）
        
        设计原理：
        - Part A: 距离减少 = 主要信号（引导靠近）
        - Part B: 朝向对齐 = 辅助信号（防止侧滑、后退）
        
        预期输出（v2 - 2倍增强）：
        - 正常飞行：+3.0/step（原+1.5）
        - 后退/侧滑：-1.0/step（原-0.5）
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
        # 🔥 增强2倍：每0.1米靠近 = +1.6分（原来0.8）
        if distance_change > 0.01:  # 靠近目标
            reward_distance = distance_change * 16.0  # 8.0 → 16.0
            reward_distance = min(reward_distance, 4.0)  # 单步最多+4（原+2）
        elif distance_change < -0.01:  # 远离目标
            reward_distance = distance_change * 16.0  # 8.0 → 16.0
            reward_distance = max(reward_distance, -2.0)  # 单步最多-2（原-1）
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
                
                # 🔥 增强2倍朝向奖励
                if alignment > 0.7:  # 朝向目标（cos(45°)≈0.7）
                    reward_alignment = 1.0 * (alignment - 0.7) / 0.3  # 0到+1.0（原+0.5）
                elif alignment < 0:  # 背对目标
                    reward_alignment = -0.6 * abs(alignment)  # 0到-0.6（原-0.3）
                # 侧向不给奖励也不惩罚（允许绕路避障）
        
        # 合并奖励
        total_reward = reward_distance + reward_alignment
        
        return total_reward
    
    def _compute_safe_navigation_reward(self, depth_info: Dict[str, float],
                                       velocity: np.ndarray,
                                       orientation: Optional[np.ndarray],
                                       distance_to_target: float) -> float:
        """🎯 融合版安全导航奖励 - 固定翼特化设计
        
        🛫 固定翼特性：
        1. 不能悬停或急刹车
        2. 避障主要靠转弯
        3. 推力减小只能缓慢降速
        4. 必须保持最小前进速度
        
        避障策略：
        - 开阔空间：全速直行
        - 发现障碍：提前转向
        - 危险区域：大角度转弯+适度降速
        - 紧急情况：急转弯逃离
        
        预期输出：
        - 开阔全速：+2.0/step
        - 提前转向：+1.5/step
        - 正确避障：+1.0/step
        - 错误行为：-1.6/step
        """
        depth_map = depth_info.get('depth_map', None)
        
        if depth_map is None:
            return 0.0
        
        # ===== Step 1: 使用深度处理器进行专业分析 =====
        obstacle_analysis = self.depth_processor.get_obstacle_analysis(depth_map)
        
        # 获取结构化的障碍物信息
        danger_level = obstacle_analysis['danger_level']  # 0-1，1最危险
        forward_openness = obstacle_analysis['forward_openness']  # 0-1，1最开阔
        physical_min_depth = obstacle_analysis['physical_min_depth']  # 物理距离（米）
        
        # 从 depth_info 获取已计算的左右区域深度（避免重复计算）
        left_depth = depth_info.get('left_min', 0.5)  # 标准化深度
        right_depth = depth_info.get('right_min', 0.5)
        
        # 计算速度和角速度
        speed_2d = np.linalg.norm(velocity[:2])
        angular_vel = depth_info.get('angular_velocity', 0.0)
        
        # ===== Step 2: 固定翼避障策略 =====
        
        # � 固定翼核心：避障靠转弯，不能减速悬停
        
        # 场景A: 非常开阔（danger_level < 0.2，前方>4m）→ 全速直行
        if danger_level < 0.2:
            # 前方安全，鼓励保持速度
            if speed_2d > 1.5:  # 保持较高速度
                return +2.0  # 优秀！全速前进
            elif speed_2d > 1.0:  # 中等速度
                return +1.2  # 不错，但可以更快
            else:  # 速度太慢
                return +0.3  # 前方开阔，应该加速
        
        # 场景B: 较开阔（danger_level < 0.4，前方2-4m）→ 保持速度，准备转向
        elif danger_level < 0.4:
            # 前方还算安全，保持速度，如果有障碍物则准备转向
            left_right_diff = abs(left_depth - right_depth) * self.depth_processor.depth_scale
            
            if left_right_diff > 0.5:  # 左右有差异，应该转向开阔方向
                should_turn_left = left_depth > right_depth
                is_turning = (should_turn_left and angular_vel < -0.02) or \
                            (not should_turn_left and angular_vel > 0.02)
                
                if is_turning:
                    # 正在提前转向，很好！
                    if speed_2d > 1.0:  # 保持速度的同时转向
                        return +1.5  # 优秀：提前规避+保持速度
                    else:
                        return +1.0  # 不错：在转向
                else:
                    # 应该转向但没转
                    if speed_2d > 1.0:
                        return +0.8  # 还行，至少保持速度
                    else:
                        return +0.3  # 一般
            else:
                # 左右差不多，直行即可
                if speed_2d > 1.0:
                    return +1.2  # 好，保持速度
                else:
                    return +0.5  # 一般
        
        # 场景C: 接近障碍（danger_level < 0.7，前方1-2m）→ 必须转向！
        elif danger_level < 0.7:
            # 🛫 固定翼关键：这里必须转弯，不能靠减速
            left_right_diff = abs(left_depth - right_depth) * self.depth_processor.depth_scale
            
            if left_right_diff > 0.3:  # 有任何左右差异就应该转向
                should_turn_left = left_depth > right_depth
                is_turning_correctly = (should_turn_left and angular_vel < -0.03) or \
                                      (not should_turn_left and angular_vel > 0.03)
                
                if is_turning_correctly:
                    # 正在转向开阔方向，固定翼的正确避障！
                    if abs(angular_vel) > 0.08:  # 大角度转弯
                        return +1.5  # 优秀：大角度避障
                    elif abs(angular_vel) > 0.04:  # 中等转弯
                        return +1.0  # 好：正在转向
                    else:  # 小角度转弯
                        return +0.6  # 还行，但转得不够
                elif abs(angular_vel) > 0.03:
                    # 转错方向了！
                    return -0.8  # 危险：转向错误方向
                else:
                    # 危险：没有转向！
                    return -1.2  # 严重错误：障碍物近了还不转
            else:
                # 两边差不多，随便选个方向转
                if abs(angular_vel) > 0.05:  # 在转弯
                    return +0.8  # 好，至少在避障
                else:
                    return -1.0  # 危险：不转弯
        
        # 场景D: 非常危险（danger_level >= 0.7，前方<1m）→ 紧急转向！
        else:
            # � 紧急情况：必须大角度急转！
            if abs(angular_vel) > 0.1:  # 大角度急转
                return +1.2  # 好！紧急避障
            elif abs(angular_vel) > 0.05:  # 中等转弯
                return +0.6  # 还行，但应该转更急
            else:
                # 非常危险还不转弯！
                return -2.0  # 严重错误：即将碰撞还不转
        
        return 0.0
    
    def reset_state(self):
        """重置状态（用于新回合）"""
        self.previous_distances.clear()


def create_default_reward_config() -> Dict[str, Any]:
    """🎯 极简版奖励配置 - 平衡稳定版 v6（添加超时惩罚）
    
    核心设计：
    1. 🔥 成功奖励占比要足够大，引导正确学习方向
    2. 🔥 碰撞惩罚确保错误行为代价明确
    3. 🔥 超时惩罚防止"拖时间"策略
    4. 🔥 密集奖励强度为1.0，提供充分导航指导
    
    实际数据分析（max_steps=500，加速训练）：
    - 500步超时的实际密集奖励：≈3500分（7分/步）
    - 理论最大密集奖励：7分/步 × 500步 = 3500分
    - 正常导航密集奖励：≈420-1400分（60-200步）
    
    奖励测算（v8 - max_steps=1000，强负值超时）：
    - 快速成功(60步):  +4500 +420 = +4920分  ✓ 效率最优
    - 慢速成功(200步): +4500 +1400 = +5900分  ✓ 最高奖励  
    - 碰撞失败(150步): -2000 +1050 = -950分  ✗ 明确负值
    - 超时失败(1000步): -8000 +7000 = -1000分 ✗✗ 强负值（比碰撞还差！）
    
    关键改进（v7→v8）：
    1. 超时惩罚: -500 → -8000（彻底解决"拖延策略"）
    2. 碰撞惩罚: -1000 → -2000（确保明确负值）
    3. 奖励排序：成功(+5900) >>> 碰撞(-950) > 超时(-1000)
    4. Agent学到："必须成功！拖延是最差策略！"
    
    设计理念：
    ✅ 快速成功最优 → 鼓励高效导航
    ✅ 慢速成功次优 → 允许谨慎探索
    ✅ 超时失为正值，但明显低于成功 → 不鼓励拖时间
    ✅ 碰撞必为负值 → 明确错误代价
    """
    return {
        # 稀疏奖励 - 🔥 强化版设计（解决log_std增长+超时问题）
        'success_bonus': 4500.0,        # 成功奖励
        'crash_penalty': -2000.0,       # 🔥 碰撞强惩罚，确保碰撞必为负值
        
        # 避障参数
        'collision_distance': 0.6,       # 碰撞阈值：0.6米
        'dense_scale': 1.0,              # 🔥 密集奖励不缩放（保持1.0）
                                         # 理由：提供充分的导航指导能力，
                                         #      超时惩罚-8000足够大，确保超时为强负值
        
        # 深度处理器参数
        'depth_scale': 4.0,              # 深度缩放因子
        'max_depth': 2.0,                # 最大深度值
        'cnn_feature_dim': 128,          # CNN特征维度
    }
