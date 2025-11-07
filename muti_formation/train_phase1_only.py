"""
第一阶段专用训练脚本：领航者避障和导航训练
包含实时奖励图表绘制功能

路径配置说明:
- 所有保存路径都通过 PathConfig 类集中管理
- 基础目录: agent/log/ (日志), agent/model/ (模型)
- 主要路径:
  * TRAINING_PROGRESS_PLOT: 训练进度图 (training_progress.png)
  * TRAINING_DATA_JSON: 训练数据 (training_data.json)
  * TRAJECTORIES_JSON: 轨迹数据 (trajectories.json)
  * FINAL_MODEL: 最终模型 (leader_phase1_final.pth)
  * FINAL_PROGRESS_PLOT: 最终进度图 (leader_phase1_final_progress.png)
  * FINAL_DATA_JSON: 最终数据 (leader_phase1_final_data.json)
  * FINAL_TRAJECTORIES_JSON: 最终轨迹 (leader_phase1_final_trajectories.json)

轨迹保存策略:
- 定期保存: 每 plot_interval 个回合自动保存最近的轨迹数据
- 内存管理: 保存后自动清理旧轨迹，只保留最近的 save_interval 个轨迹
- 最终保存: 训练结束时保存完整的轨迹数据到最终文件
- 防止内存溢出: 避免一次性保存大量轨迹数据导致的问题
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from collections import deque
import time
import pybullet as p

from agent.CTDE_PPO_agent import CTDE_PPO, device
from drone_envs.envs.drone_env_multi import DroneNavigationMulti

class PathConfig:
    """路径配置类 - 集中管理所有保存路径"""
    
    # 基础目录
    BASE_DIR = Path(__file__).parent
    AGENT_DIR = BASE_DIR / "agent"
    LOG_DIR = AGENT_DIR / "log"
    MODEL_DIR = AGENT_DIR / "model"
    
    # 训练进度相关路径
    TRAINING_PROGRESS_PLOT = LOG_DIR / "training_progress.png"
    TRAINING_DATA_JSON = LOG_DIR / "training_data.json"
    TRAJECTORIES_JSON = LOG_DIR / "trajectories.json"
    
    # 最终结果路径
    FINAL_MODEL = MODEL_DIR / "leader_phase1_final.pth"
    FINAL_PROGRESS_PLOT = LOG_DIR / "leader_phase1_final_progress.png"
    FINAL_DATA_JSON = LOG_DIR / "leader_phase1_final_data.json"
    FINAL_TRAJECTORIES_JSON = LOG_DIR / "leader_phase1_final_trajectories.json"
    
    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录存在"""
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_episode_model_path(cls, episode_num):
        """获取指定回合的模型保存路径"""
        return cls.MODEL_DIR / f"leader_phase1_episode_{episode_num}.pth"
    
    @classmethod
    def get_timestamped_path(cls, base_name, extension="json"):
        """获取带时间戳的路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls.LOG_DIR / f"{base_name}_{timestamp}.{extension}"

class RewardTracker:
    """奖励跟踪和可视化类"""
    def __init__(self, window_size=100):
        self.episode_rewards = []
        self.episode_lengths = []
        self.moving_avg_rewards = []
        self.success_rate = []
        self.collision_rate = []
        self.window_size = window_size
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置图表内存警告阈值，避免警告干扰
        plt.rcParams['figure.max_open_warning'] = 50
        
    def add_episode(self, episode_reward, episode_length, success, collision):
        """添加新回合数据"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # 计算移动平均
        if len(self.episode_rewards) >= self.window_size:
            moving_avg = np.mean(self.episode_rewards[-self.window_size:])
        else:
            moving_avg = np.mean(self.episode_rewards)
        self.moving_avg_rewards.append(moving_avg)
        
        # 存储成功标志
        if not hasattr(self, 'success_flags'):
            self.success_flags = []
        self.success_flags.append(success)
        
        # 计算成功率 - 使用存储的success标志
        recent_episodes = min(len(self.success_flags), self.window_size)
        recent_successes = sum(self.success_flags[-recent_episodes:])
        self.success_rate.append(recent_successes / recent_episodes)
        
        # 碰撞率
        self.collision_rate.append(1.0 if collision else 0.0)
        
    def plot_training_progress(self, save_path=None):
        """绘制训练进度图"""
        if save_path is None:
            save_path = PathConfig.TRAINING_PROGRESS_PLOT
        
        # 关闭之前的图表，避免内存泄漏
        plt.close('all')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        episodes = range(1, len(self.episode_rewards) + 1)
        
        # 1. 奖励曲线
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='原始奖励')
        ax1.plot(episodes, self.moving_avg_rewards, color='red', linewidth=2, label=f'{self.window_size}回合移动平均')
        ax1.set_xlabel('回合数')
        ax1.set_ylabel('奖励')
        ax1.set_title('领航者训练奖励曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 回合长度
        ax2.plot(episodes, self.episode_lengths, color='green', alpha=0.7)
        ax2.set_xlabel('回合数')
        ax2.set_ylabel('回合长度（步数）')
        ax2.set_title('回合长度变化')
        ax2.grid(True, alpha=0.3)
        
        # 3. 碰撞率
        ax3.plot(episodes, self.collision_rate, color='red', alpha=0.7)
        ax3.set_xlabel('回合数')
        ax3.set_ylabel('碰撞率')
        ax3.set_title('碰撞率变化')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. 成功率
        ax4.plot(episodes, self.success_rate, color='purple', alpha=0.7)
        ax4.set_xlabel('回合数')
        ax4.set_ylabel('成功率')
        ax4.set_title(f'导航成功率 ({self.window_size}回合滑动窗口)')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练进度图已保存: {save_path}")
        
        # 注意：不在这里关闭图表，让图表可以显示
        # 下次调用 plot_training_progress 时会通过 plt.close('all') 自动关闭
        return fig
    
    def save_data(self, save_path=None):
        """保存训练数据"""
        if save_path is None:
            save_path = PathConfig.TRAINING_DATA_JSON
        # 将numpy类型转换为Python原生类型以支持JSON序列化
        data = {
            'episode_rewards': [float(x) for x in self.episode_rewards],
            'episode_lengths': [int(x) for x in self.episode_lengths],
            'moving_avg_rewards': [float(x) for x in self.moving_avg_rewards],
            'success_rate': [float(x) for x in self.success_rate],
            'success_flags': [bool(x) for x in getattr(self, 'success_flags', [])],
            'collision_rate': [float(x) for x in self.collision_rate],
            'total_episodes': len(self.episode_rewards),
            'final_avg_reward': float(self.moving_avg_rewards[-1] if self.moving_avg_rewards else 0),
            'final_success_rate': float(self.success_rate[-1] if self.success_rate else 0),
            'timestamp': str(datetime.now())
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"训练数据已保存: {save_path}")

class TrajectoryTracker:
    """轨迹跟踪和记录类 - 支持定期保存以避免内存问题"""
    
    def __init__(self, save_interval=100, auto_save=True):
        self.trajectories = []  # 存储所有回合的轨迹数据
        self.save_interval = save_interval  # 每多少个回合保存一次
        self.auto_save = auto_save  # 是否启用自动保存
        self.last_save_episode = 0  # 上次保存的回合数
        
    def start_episode(self, episode_id, goal_position, start_position, num_drones):
        """开始新回合的轨迹记录"""
        episode_data = {
            'episode_id': episode_id,
            'goal_position': goal_position.tolist() if hasattr(goal_position, 'tolist') else goal_position,
            'start_position': start_position.tolist() if hasattr(start_position, 'tolist') else start_position,
            'num_drones': num_drones,
            'drone_trajectories': [[] for _ in range(num_drones)],  # 每个无人机的轨迹点
            'timestamps': [],  # 时间戳
            'rewards': [],     # 每步奖励
            'actions': [],     # 每步动作
            'success': False,
            'collision': False,
            'termination_reason': 'unknown',
            'total_steps': 0,
            'total_reward': 0.0,
            'environment_info': {}  # 环境相关信息
        }
        return episode_data
    
    def record_step(self, episode_data, drone_positions, timestamp, reward, action):
        """记录每一步的轨迹信息 - 只记录二维平面坐标"""
        episode_data['timestamps'].append(timestamp)
        episode_data['rewards'].append(float(reward))
        episode_data['actions'].append(action.tolist() if hasattr(action, 'tolist') else action)
        
        # 只记录二维平面坐标 (x, y)，忽略z坐标
        for i, pos in enumerate(drone_positions):
            if i < len(episode_data['drone_trajectories']):
                # 只保存x, y坐标，适用于平面模式
                plane_pos = [float(pos[0]), float(pos[1])]  # x, y坐标
                episode_data['drone_trajectories'][i].append(plane_pos)
    
    def end_episode(self, episode_data, success, collision, termination_reason, total_reward, total_steps, environment_info=None):
        """结束回合记录"""
        episode_data['success'] = success
        episode_data['collision'] = collision
        episode_data['termination_reason'] = termination_reason
        episode_data['total_reward'] = float(total_reward)
        episode_data['total_steps'] = total_steps
        
        if environment_info:
            episode_data['environment_info'] = environment_info
            
        # 确保所有轨迹长度一致
        min_length = min(len(traj) for traj in episode_data['drone_trajectories']) if episode_data['drone_trajectories'] else 0
        for i in range(len(episode_data['drone_trajectories'])):
            episode_data['drone_trajectories'][i] = episode_data['drone_trajectories'][i][:min_length]
        
        # 截断其他列表以保持一致性
        episode_data['timestamps'] = episode_data['timestamps'][:min_length]
        episode_data['rewards'] = episode_data['rewards'][:min_length]
        episode_data['actions'] = episode_data['actions'][:min_length]
        
        self.trajectories.append(episode_data)
        return episode_data
    
    def periodic_save(self, current_episode, save_path=None):
        """定期保存轨迹数据 - 只保存最近的轨迹，避免内存累积"""
        if not self.auto_save:
            return
            
        if current_episode - self.last_save_episode >= self.save_interval:
            try:
                # 创建临时保存路径
                if save_path is None:
                    base_path = PathConfig.TRAJECTORIES_JSON
                    temp_path = base_path.parent / f"{base_path.stem}_ep{current_episode}{base_path.suffix}"
                else:
                    temp_path = Path(save_path)
                
                # 只保存最近的轨迹数据，避免文件过大
                recent_trajectories = self.trajectories[-self.save_interval:] if len(self.trajectories) > self.save_interval else self.trajectories
                
                self._save_trajectories_to_file(recent_trajectories, temp_path)
                print(f"定期轨迹保存: {temp_path} (最近{len(recent_trajectories)}个回合)")
                
                # 清理内存：只保留最近的轨迹数据
                if len(self.trajectories) > self.save_interval:
                    # 保留最近的save_interval个轨迹，用于下次保存
                    self.trajectories = self.trajectories[-self.save_interval:]
                    print(f"内存清理：保留最近{len(self.trajectories)}个轨迹")
                
                self.last_save_episode = current_episode
                
            except Exception as e:
                print(f"定期轨迹保存失败: {e}")
                import traceback
                traceback.print_exc()
    
    def save_trajectories(self, save_path=None):
        """保存所有轨迹数据到文件"""
        if save_path is None:
            save_path = PathConfig.TRAJECTORIES_JSON
        
        return self._save_trajectories_to_file(self.trajectories, save_path)
    
    def _save_trajectories_to_file(self, trajectories_data, save_path):
        """内部方法：将轨迹数据保存到文件"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 自定义JSON编码器，处理numpy类型
        def numpy_encoder(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [numpy_encoder(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: numpy_encoder(value) for key, value in obj.items()}
            else:
                return obj
        
        # 转换为JSON可序列化格式，并处理numpy类型
        serializable_data = {
            'total_episodes': len(trajectories_data),
            'trajectories': numpy_encoder(trajectories_data),
            'summary': {
                'successful_episodes': sum(1 for t in trajectories_data if t['success']),
                'collision_episodes': sum(1 for t in trajectories_data if t['collision']),
                'average_reward': float(np.mean([t['total_reward'] for t in trajectories_data])) if trajectories_data else 0.0,
                'average_steps': float(np.mean([t['total_steps'] for t in trajectories_data])) if trajectories_data else 0.0,
                'timestamp': str(datetime.now())
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"轨迹数据已保存: {save_path} (共{len(trajectories_data)}个回合)")
        
        return save_path

def train_leader_phase1(max_episodes=20000, max_steps=3000, plot_interval=1000):
    """第一阶段：专门训练领航者避障和导航
    
    Args:
        max_episodes: 最大训练回合数，20000回合用于充分学习
        max_steps: 每回合最大步数，3000步，给无人机充足时间到达目标
        plot_interval: 绘图间隔，1000回合保存一次
    """
    print("="*80)
    print("开始第一阶段训练：领航者避障和导航")
    print("="*80)
    
    # 确保所有目录存在
    PathConfig.ensure_directories()
    
    # 创建环境
    env = DroneNavigationMulti(
        num_drones=1,  # 修改为仅1架领航者
        use_depth_camera=True,
        depth_camera_range=10.0,
        depth_resolution=16,
        enable_formation_force=False,  # 关闭编队力
        training_stage=1,  # 第一阶段
        max_steps=max_steps  # 传递最大步数参数
    )
    
    print(f"环境配置:")
    print(f"  - 无人机数量: {env.num_drones}")
    print(f"  - 观测空间: {env.observation_space.shape}")
    print(f"  - 深度特征维度: {env.depth_feature_dim}")
    print(f"  - 训练阶段: {env.training_stage}")
    print(f"  - 编队力状态: {'禁用' if not env.enable_formation_force else '启用'}")
    print(f"  - 平面模式: {'启用' if env.enforce_planar else '禁用'}")
    
    # 创建CTDE代理 - 根据平面模式调整状态维度
    # 平面模式：位置(2) + 速度(2) + 朝向(4) + 目标相对位置(2) + 深度特征(130) = 140维
    leader_state_dim = 2 + 2 + 4 + 2 + env.depth_feature_dim if env.enforce_planar else 3 + 3 + 4 + 3 + env.depth_feature_dim
    # 位置(2/3) + 速度(2/3) + 朝向(4) + 目标相对位置(2/3) + 深度特征(130)
    
    ppo_agent = CTDE_PPO(
        leader_state_dim=leader_state_dim,
        follower_state_dim=leader_state_dim,  # 跟随者状态维度与领航者相同
        leader_visual_dim=env.depth_feature_dim,
        action_dim=2,  # 前进/后退力和转向扭矩
        num_drones=1,  # 第一阶段只训练领航者
        lr_actor=0.0003,   # 🔧 从0.0005降低到0.0003，提高训练稳定性，防止覆盖成功策略
        lr_critic=0.001,   # Critic学习率
        gamma=0.99,        # 折扣因子
        K_epochs=40,       # 🎯 PPO更新轮次，从20恢复到40以充分学习
        eps_clip=0.2,      # PPO裁剪参数
        has_continuous_action_space=True,
        action_std_init=0.3  # 🎯 20000回合优化: 从0.1增加到0.3，保持长期探索能力
    )
    
    print("CTDE代理已创建")
    print("="*80)
    
    # 初始化奖励跟踪器和轨迹跟踪器
    reward_tracker = RewardTracker(window_size=50)
    trajectory_tracker = TrajectoryTracker(save_interval=plot_interval, auto_save=True)  # 定期保存间隔与绘图间隔相同
    
    # 训练循环
    start_time = time.time()
    
    for episode in range(max_episodes):
        # 确保buffer在episode开始时是空的
        for buffer in ppo_agent.buffers:
            buffer.clear()
            
        state, _ = env.reset()
        
        # 初始化轨迹记录
        leader_start_pos = np.array(env.start_position)
        goal_pos = np.array(env.goal)
        episode_trajectory = trajectory_tracker.start_episode(
            episode_id=episode + 1,
            goal_position=goal_pos,
            start_position=leader_start_pos,
            num_drones=env.num_drones
        )
        
        episode_reward = 0
        leader_reward_total = 0
        obstacle_detections = 0
        collision_occurred = False
        collision_type = ""  # 碰撞类型
        
        for step in range(max_steps):
            # 提取领航者观测 - 根据平面模式调整深度特征位置
            leader_obs_dim = 2 + 2 + 4 + 2 + env.depth_feature_dim if env.enforce_planar else 3 + 3 + 4 + 3 + env.depth_feature_dim
            leader_obs = state[:leader_obs_dim]
            # 在平面模式下，深度特征从索引10开始；在3D模式下从索引13开始
            depth_start_idx = 2 + 2 + 4 + 2 if env.enforce_planar else 3 + 3 + 4 + 3
            depth_features = leader_obs[depth_start_idx:depth_start_idx + env.depth_feature_dim] if env.use_leader_camera else None
            
            # 监控避障信息
            if hasattr(env, 'depth_obstacle_processor') and env.use_leader_camera:
                try:
                    # 使用屏蔽后的深度图像进行避障检测，避免无人机自身被误认为障碍物
                    depth_image = env._get_masked_leader_depth()
                    if depth_image is not None and depth_image.size > 0:
                        raw_depth = depth_image if len(depth_image.shape) == 2 else depth_image[:, :, 0]
                        processed_depth = env.depth_obstacle_processor.preprocess_depth_image(raw_depth)
                        obstacle_detected, min_depth = env.depth_obstacle_processor.detect_obstacles(processed_depth)
                        
                        if obstacle_detected:
                            obstacle_detections += 1
                        
                        # 移除深度碰撞检测，避免与环境碰撞检测冲突
                        # 碰撞检测由环境统一处理，这里只用于统计避障信息
                            
                except Exception as e:
                    pass
            
            # 第一阶段：只控制领航者
            leader_action = ppo_agent.select_action([leader_obs], depth_features)[0]
            
            # 环境步进 - 第一阶段直接使用领航者动作
            next_state, reward, terminated, truncated, info = env.step(leader_action)
            
            # 记录轨迹信息
            current_time = time.time()
            # 获取当前所有无人机的位置
            drone_positions = []
            for i in range(env.num_drones):
                pos, _ = p.getBasePositionAndOrientation(env.drones[i].drone, env.client)
                drone_positions.append(np.array(pos))
            
            trajectory_tracker.record_step(
                episode_trajectory, 
                drone_positions, 
                current_time, 
                reward, 
                leader_action
            )
            
            # 记录奖励和碰撞信息 - 使用环境返回的奖励信息
            reward_info = info.get('reward_info', {})
            episode_reward += reward  # 使用环境返回的总奖励
            
            # 🎯 奖励监控：修复后success=3000是正常值
            # 单步奖励范围: 成功步 ~3010 (3000+密集), 普通步 ~10, 失败步 ~-100
            if abs(reward) > 3500:  # 🔧 阈值从1000提升到3500（success=3000 + 余量）
                # 只在异常高的奖励时警告（理论最大值应该不超过3100）
                print(f"⚠️  检测到异常奖励值 {reward:.2f} 在步数 {step + 1}")
                print(f"  奖励详情: {reward_info}")
            elif reward > 3000:  # 成功奖励，记录但不警告
                print(f"🎉 成功！奖励 {reward:.2f} 在步数 {step + 1}")
                # 移除截断：环境中的reward_calculator已经正确处理奖励范围
            
            # 从环境获取碰撞信息 - 只在真正终止（不是截断）时记录碰撞
            crash_reward = reward_info.get('crash', 0)
            if crash_reward < 0 and terminated:  # 只有真正碰撞终止时才记录碰撞
                collision_occurred = True
                # 从深度信息中获取碰撞类型（如果可获取）
                if hasattr(env, '_get_depth_info'):
                    try:
                        depth_info = env._get_depth_info()
                        collision_type = depth_info.get('collision_type', 'unknown')
                        if collision_type == 'physical_contact':
                            collision_type = f"物理碰撞({depth_info.get('contact_points', 0)}点)"
                        elif collision_type == 'boundary':
                            collision_type = "边界碰撞"
                        else:
                            collision_type = "深度碰撞"
                    except:
                        collision_type = "碰撞"
            
            # 存储经验
            ppo_agent.buffers[0].rewards.append(reward)
            ppo_agent.buffers[0].is_terminals.append(terminated or truncated)
            
            state = next_state
            
            if terminated or truncated:
                break
        
        # 更新策略
        if len(ppo_agent.buffers[0].rewards) > 0:
            try:
                ppo_agent.update()
            except Exception as e:
                print(f"策略更新失败: {e}")
                # 即使更新失败也要清空buffer，避免奖励累积
                for buffer in ppo_agent.buffers:
                    buffer.clear()
                continue
        
        # 计算统计信息
        success = info.get('success', False)  # 从环境获取真正的成功标志
        
        # 根据终止类型设置显示信息
        if terminated and not success:
            # 真正碰撞终止
            termination_type = collision_type if collision_occurred else "终止"
        elif truncated:
            # 达到最大步数截断
            termination_type = "超时"
        elif success:
            # 成功到达
            termination_type = "成功"
        else:
            termination_type = "未知"
        
        # 记录数据
        reward_tracker.add_episode(episode_reward, step + 1, success, collision_occurred)
        
        # 结束轨迹记录
        trajectory_tracker.end_episode(
            episode_trajectory, 
            success, 
            collision_occurred, 
            termination_type, 
            episode_reward, 
            step + 1, 
            {'reward_info': info.get('reward_info', {})}
        )
        
        # 输出进度
        elapsed_time = time.time() - start_time
        eta = (elapsed_time / (episode + 1)) * (max_episodes - episode - 1)

        # 计算当前统计信息
        current_avg_reward = np.mean(reward_tracker.episode_rewards[-10:]) if len(reward_tracker.episode_rewards) >= 10 else episode_reward
        current_success_rate = reward_tracker.success_rate[-1] if reward_tracker.success_rate else 0
        current_collision_rate = np.mean(reward_tracker.collision_rate[-10:]) if len(reward_tracker.collision_rate) >= 10 else (1.0 if collision_occurred else 0.0)

        print(f"回合 {episode + 1:3d}/{max_episodes:3d} | "
              f"奖励: {episode_reward:8.2f} | "
              f"平均: {current_avg_reward:6.2f} | "
              f"步数: {step + 1:4d} | "
              f"成功: {'✓' if success else '✗'} | "
              f"成功率: {current_success_rate:.1%} | "
              f"终止: {termination_type} | "
              f"碰撞率: {current_collision_rate:.1%} | "
              f"ETA: {eta/60:.1f}分钟")
        
        # 定期绘制和保存
        if (episode + 1) % plot_interval == 0:
            reward_tracker.plot_training_progress()
            reward_tracker.save_data()
            
            # 定期保存轨迹数据
            trajectory_tracker.periodic_save(episode + 1)
            
            # 保存当前模型
            model_path = PathConfig.get_episode_model_path(episode + 1)
            ppo_agent.save(model_path)
            print(f"模型已保存: {model_path}")
            
            # 🔥 显示经验回放缓冲区统计信息
            replay_stats = ppo_agent.get_replay_buffer_stats()
            print(f"📦 经验回放缓冲区: "
                  f"容量 {replay_stats['size']}/1000 | "  # 🔥 更新显示为1000
                  f"成功经验 {replay_stats['success_count']} ({replay_stats['success_rate']:.1%}) | "
                  f"累计添加 {replay_stats['total_added']}")
            
            # 🎯 修复探索策略: 延长探索期，防止过早收敛到局部最优
            # 阶段1 (0-8000): 高探索 0.3 → 0.25 (温和衰减)
            # 阶段2 (8000-16000): 中探索 0.25 → 0.15 (适度衰减)
            # 阶段3 (16000-20000): 低探索 0.15 → 0.08 (保留探索)
            if episode + 1 <= 8000:
                # 前8000回合：每3000回合衰减一次，保持高探索
                if (episode + 1) % 3000 == 0:
                    current_std = ppo_agent.action_std
                    ppo_agent.decay_action_std(action_std_decay_rate=0.99, min_action_std=0.25)
                    print(f"🔍 探索衰减: {current_std:.4f} → {ppo_agent.action_std:.4f} (阶段1: 高探索期)")
            elif episode + 1 <= 16000:
                # 中期8000-16000：每3000回合衰减一次
                if (episode + 1) % 3000 == 0:
                    current_std = ppo_agent.action_std
                    ppo_agent.decay_action_std(action_std_decay_rate=0.97, min_action_std=0.15)
                    print(f"🔍 探索衰减: {current_std:.4f} → {ppo_agent.action_std:.4f} (阶段2: 中探索期)")
            else:
                # 后期16000-20000：每2000回合衰减一次，保留足够探索
                if (episode + 1) % 2000 == 0:
                    current_std = ppo_agent.action_std
                    ppo_agent.decay_action_std(action_std_decay_rate=0.96, min_action_std=0.08)
                    print(f"🔍 探索衰减: {current_std:.4f} → {ppo_agent.action_std:.4f} (阶段3: 精调期)")
            
            print("-" * 80)
    
    # 训练完成
    print("="*80)
    print("第一阶段训练完成！")
    
    # 最终统计
    final_avg_reward = np.mean(reward_tracker.episode_rewards[-50:]) if len(reward_tracker.episode_rewards) >= 50 else np.mean(reward_tracker.episode_rewards)
    final_success_rate = reward_tracker.success_rate[-1] if reward_tracker.success_rate else 0
    
    print(f"最终统计:")
    print(f"  - 总回合数: {len(reward_tracker.episode_rewards)}")
    print(f"  - 最终平均奖励: {final_avg_reward:.2f}")
    print(f"  - 最终成功率: {final_success_rate:.2%}")
    print(f"  - 训练时长: {(time.time() - start_time)/60:.1f}分钟")
    
    # 保存最终模型和数据
    final_model_path = PathConfig.FINAL_MODEL
    ppo_agent.save(final_model_path)
    
    reward_tracker.plot_training_progress(PathConfig.FINAL_PROGRESS_PLOT)
    reward_tracker.save_data(PathConfig.FINAL_DATA_JSON)
    
    # 保存完整的轨迹数据（合并所有定期保存的数据）
    try:
        trajectory_tracker.save_trajectories(PathConfig.FINAL_TRAJECTORIES_JSON)
        print(f"完整轨迹数据已保存: {PathConfig.FINAL_TRAJECTORIES_JSON}")
    except Exception as e:
        print(f"完整轨迹保存失败: {e}")
        # 如果完整保存失败，尝试保存当前内存中的数据
        try:
            backup_path = PathConfig.FINAL_TRAJECTORIES_JSON.parent / f"{PathConfig.FINAL_TRAJECTORIES_JSON.stem}_backup{PathConfig.FINAL_TRAJECTORIES_JSON.suffix}"
            trajectory_tracker._save_trajectories_to_file(trajectory_tracker.trajectories, backup_path)
            print(f"备份轨迹数据已保存: {backup_path}")
        except Exception as e2:
            print(f"备份轨迹保存也失败: {e2}")
    
    print(f"最终模型已保存: {final_model_path}")
    print("="*80)
    
    env.close()
    return final_model_path, reward_tracker, trajectory_tracker

if __name__ == '__main__':
    # 运行第一阶段训练
    model_path, reward_tracker, trajectory_tracker = train_leader_phase1(
        max_episodes=20000,  # 从500增加到2000回合
        max_steps=3000,   # 每回合最大步数，给无人机充足时间到达目标
        plot_interval=1000   # 每100个回合绘制一次图
    )
    
    print(f"训练完成！检查 {PathConfig.LOG_DIR} 目录查看结果图表和数据。")
