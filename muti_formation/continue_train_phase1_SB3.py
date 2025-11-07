"""
继续训练脚本 - 第一阶段领航者单机导航训练

训练目标:
- 继续训练领航者无人机进行单机导航
- 基于之前训练的模型继续学习
- 凑齐100000回合训练
- 使用PPO算法进行强化学习训练
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

# 导入stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import random
from drone_envs.envs.drone_env_multi import DroneNavigationMulti

class PathConfig:
    """路径配置类 - 集中管理所有保存路径 (SB3专用)"""
    
    # 基础目录 - 放在agent目录下
    BASE_DIR = Path(__file__).parent
    AGENT_DIR = BASE_DIR / "agent"
    LOG_SB3_DIR = AGENT_DIR / "log_SB3"
    MODEL_SB3_DIR = AGENT_DIR / "model_SB3"
    
    # 简化引用
    LOG_DIR = LOG_SB3_DIR
    MODEL_DIR = MODEL_SB3_DIR
    
    # 训练进度相关路径
    TRAINING_PROGRESS_PLOT = LOG_DIR / "training_progress.png"
    TRAINING_DATA_JSON = LOG_DIR / "training_data.json"
    TRAJECTORIES_JSON = LOG_DIR / "trajectories.json"
    
    # 最终结果路径
    FINAL_MODEL = MODEL_DIR / "leader_phase1_final"
    FINAL_PROGRESS_PLOT = LOG_DIR / "leader_phase1_final_progress.png"
    FINAL_DATA_JSON = LOG_DIR / "leader_phase1_final_data.json"
    FINAL_TRAJECTORIES_JSON = LOG_DIR / "leader_phase1_final_trajectories.json"
    
    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录存在"""
        cls.AGENT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_SB3_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_SB3_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_episode_model_path(cls, episode_num):
        """获取指定回合的模型保存路径"""
        return cls.MODEL_DIR / f"leader_phase1_episode_{episode_num}"
    
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
        ax1.set_title('第一阶段训练奖励曲线')
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
        print(f"✅ 训练进度图已保存: {save_path}")
        
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
            'algorithm': 'Stable-Baselines3 PPO',
            'timestamp': str(datetime.now())
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ 训练数据已保存: {save_path}")

class TrainingCallback(BaseCallback):
    """自定义回调函数 - 用于跟踪训练进度和定期保存"""
    
    def __init__(self, reward_tracker, max_episodes, previous_episode_count=0, plot_interval=500, save_interval=500, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.reward_tracker = reward_tracker
        self.max_episodes = max_episodes  # 最大回合数限制
        self.previous_episode_count = previous_episode_count  # 之前的回合数
        self.plot_interval = plot_interval
        self.save_interval = save_interval
        self.episode_count = previous_episode_count  # 从之前的回合数开始
        self.episode_reward = 0
        self.episode_length = 0
        self.start_time = time.time()
        
        
    def _on_step(self) -> bool:
        """每步调用"""
        # 累积奖励
        self.episode_reward += self.locals['rewards'][0]
        self.episode_length += 1
        
        # 检查是否回合结束
        if self.locals['dones'][0]:
            self.episode_count += 1
            
            # 获取info信息
            info = self.locals['infos'][0]
            success = info.get('success', False)
            
            # 从reward_info中判断是否碰撞
            reward_info = info.get('reward_info', {})
            crash_reward = reward_info.get('crash', 0)
            collision = crash_reward < 0  # 如果有碰撞惩罚，说明发生了碰撞
            
            # 记录到tracker
            self.reward_tracker.add_episode(
                self.episode_reward,
                self.episode_length,
                success,
                collision
            )
            
            # 计算统计信息
            current_avg_reward = np.mean(self.reward_tracker.episode_rewards[-10:]) if len(self.reward_tracker.episode_rewards) >= 10 else self.episode_reward
            current_success_rate = self.reward_tracker.success_rate[-1] if self.reward_tracker.success_rate else 0
            current_collision_rate = np.mean(self.reward_tracker.collision_rate[-10:]) if len(self.reward_tracker.collision_rate) >= 10 else (1.0 if collision else 0.0)
            
            # 计算ETA
            elapsed_time = time.time() - self.start_time
            # 使用max_episodes作为总回合数
            total_episodes = self.max_episodes
            eta = (elapsed_time / (self.episode_count - self.previous_episode_count)) * (total_episodes - self.episode_count) if (self.episode_count - self.previous_episode_count) > 0 else 0
            
            # 获取终止类型
            if success:
                termination_type = "成功"
            elif collision:
                # 尝试从reward_info获取更详细的碰撞类型
                contact_points = reward_info.get('contact_points', 0)
                if contact_points > 0:
                    termination_type = f"物理碰撞({contact_points}点)"
                else:
                    termination_type = "碰撞"
            else:
                termination_type = "超时"
            
            # 打印进度
            print(f"回合 {self.episode_count:5d}/{total_episodes} | "
                  f"奖励: {self.episode_reward:8.2f} | "
                  f"平均: {current_avg_reward:6.2f} | "
                  f"步数: {self.episode_length:4d} | "
                  f"成功: {'✓' if success else '✗'} | "
                  f"成功率: {current_success_rate:.1%} | "
                  f"终止: {termination_type} | "
                  f"碰撞率: {current_collision_rate:.1%} | "
                  f"ETA: {eta/60:.1f}分钟")
            
            
            # 定期绘制和保存
            if self.episode_count % self.plot_interval == 0:
                self.reward_tracker.plot_training_progress()
                self.reward_tracker.save_data()
                
            
            # 定期保存模型
            if self.episode_count % self.save_interval == 0:
                model_path = PathConfig.get_episode_model_path(self.episode_count)
                self.model.save(model_path)
                print(f"✅ 模型已保存: {model_path}")
                print("-" * 80)
            
            # 重置回合统计
            self.episode_reward = 0
            self.episode_length = 0
            
            # 检查是否达到最大回合数，如果是则停止训练
            if self.episode_count >= self.max_episodes:
                print("="*80)
                print(f"✅ 已完成 {self.max_episodes} 回合训练，停止训练")
                print("="*80)
                return False  # 返回False停止训练
        
        return True
    

def make_env(max_steps=1000):
    """创建环境的工厂函数
    
    Args:
        max_steps: 每个回合的最大步数 (默认1000)
    """
    # 固定全局随机种子
    random.seed(42)
    np.random.seed(42)
    
    def _init():
        env = DroneNavigationMulti(
            num_drones=1,
            use_depth_camera=True,
            depth_camera_range=10.0,
            depth_resolution=16,
            enable_formation_force=False,
            training_stage=1,
            max_steps=max_steps
        )
        return env
    return _init


def continue_train_leader_phase1_sb3(target_total_episodes=100000, total_timesteps=None, plot_interval=100, 
                                    load_model_path=None):
    """继续第一阶段训练 - 领航者单机导航训练（使用PPO算法）
    
    Args:
        target_total_episodes: 目标总训练回合数 (默认100000)
        total_timesteps: 总训练步数（如果为None，则根据剩余回合数估算）
        plot_interval: 绘图和保存间隔
        load_model_path: 要加载的模型路径（如果为None，则自动查找最新模型）
    """
    print("="*80)
    print("继续第一阶段训练 - 领航者导航训练")
    print(f"目标总回合数: {target_total_episodes}")
    print("="*80)
    
    # 确保所有目录存在
    PathConfig.ensure_directories()
    
    # 创建向量化环境
    print("正在创建环境...")
    env = DummyVecEnv([make_env(max_steps=1000)])
    
    # 获取环境配置信息
    test_env = env.envs[0]
    print(f"环境配置:")
    print(f"  - 无人机数量: {test_env.num_drones}")
    print(f"  - 观测空间: {test_env.observation_space.shape}")
    print(f"  - 动作空间: {test_env.action_space.shape}")
    print(f"  - 深度特征维度: {test_env.depth_feature_dim}")
    print(f"  - 训练阶段: {test_env.training_stage}")
    print(f"  - 编队力状态: {'禁用' if not test_env.enable_formation_force else '启用'}")
    print(f"  - 平面模式: {'启用' if test_env.enforce_planar else '禁用'}")
    print(f"  - 最大步数: {test_env.max_steps}")
    
    # 检测并强制使用GPU
    if torch.cuda.is_available():
        device_name = 'cuda'
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device_name = 'cpu'
        print("⚠️  GPU不可用，使用CPU训练（速度较慢）")
    
    # 确定要加载的模型路径
    if load_model_path is None:
        # 自动查找最新的模型
        if PathConfig.FINAL_MODEL.with_suffix('.zip').exists():
            load_model_path = PathConfig.FINAL_MODEL
            print(f"🔄 找到最终模型，加载: {load_model_path}")
        else:
            # 查找最新的episode模型
            model_files = list(PathConfig.MODEL_DIR.glob("leader_phase1_episode_*.zip"))
            if model_files:
                # 按episode数排序，取最大的
                model_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
                load_model_path = model_files[-1].with_suffix('')
                print(f"🔄 找到最新episode模型，加载: {load_model_path}")
            else:
                raise FileNotFoundError("❌ 未找到任何已保存模型，无法继续训练")
    else:
        print(f"🔄 加载指定模型: {load_model_path}")
    
    # 加载模型
    print("正在加载模型...")
    model = PPO.load(load_model_path, env=env, device=device_name)
    print("✅ 模型加载成功！")
    
    # 尝试加载训练历史数据
    history_data = None
    if PathConfig.TRAINING_DATA_JSON.exists():
        print("正在加载训练历史...")
        with open(PathConfig.TRAINING_DATA_JSON, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        previous_total_episodes = history_data.get('total_episodes', 0)
        print(f"✅ 已加载 {previous_total_episodes} 回合的历史数据")
        print(f"   上次最终成功率: {history_data.get('final_success_rate', 0):.1%}")
        print(f"   上次平均奖励: {history_data.get('final_avg_reward', 0):.2f}")
    else:
        print("⚠️  未找到训练历史数据，将从0开始统计")
        previous_total_episodes = 0
    
    # 计算剩余回合数
    remaining_episodes = target_total_episodes - previous_total_episodes
    if remaining_episodes <= 0:
        print(f"⚠️  已经达到或超过目标回合数 {target_total_episodes}，当前 {previous_total_episodes} 回合")
        return None, None, None
    
    print(f"📊 继续训练计划:")
    print(f"   之前回合数: {previous_total_episodes}")
    print(f"   目标总回合数: {target_total_episodes}")
    print(f"   剩余回合数: {remaining_episodes}")
    
    # 估算总步数（如果未指定）
    if total_timesteps is None:
        # 使用更保守的估计，确保不会因为步数限制而过早停止
        # 假设平均每回合300步（考虑到探索和失败的情况）
        avg_steps_per_episode = 300
        total_timesteps = remaining_episodes * avg_steps_per_episode
        print(f"   估算总步数: {total_timesteps:,} ({remaining_episodes}回合 × {avg_steps_per_episode}步)")
        print(f"   ⚠️  注意: 实际训练将在达到 {target_total_episodes} 总回合时停止（由回调函数控制）")
    
    print("="*80)
    
    # 创建奖励跟踪器
    reward_tracker = RewardTracker(window_size=50)
    
    # 恢复历史数据
    if history_data:
        print("正在恢复训练历史...")
        reward_tracker.episode_rewards = history_data.get('episode_rewards', [])
        reward_tracker.episode_lengths = history_data.get('episode_lengths', [])
        reward_tracker.moving_avg_rewards = history_data.get('moving_avg_rewards', [])
        reward_tracker.success_rate = history_data.get('success_rate', [])
        reward_tracker.success_flags = history_data.get('success_flags', [])
        reward_tracker.collision_rate = history_data.get('collision_rate', [])
        print(f"✅ 已恢复 {len(reward_tracker.episode_rewards)} 回合的训练历史")
    
    # 创建回调函数
    callback = TrainingCallback(
        reward_tracker=reward_tracker,
        max_episodes=target_total_episodes,  # 目标总回合数
        previous_episode_count=previous_total_episodes,  # 之前的回合数
        plot_interval=plot_interval,
        save_interval=plot_interval
    )
    
    # 开始训练
    print("开始继续训练...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,  # 每10个episode打印一次日志
            progress_bar=False  # 我们使用自定义进度显示
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断！")
    
    # 训练完成
    training_time = time.time() - start_time
    print("="*80)
    print("继续训练完成！")
    
    # 最终统计
    final_avg_reward = np.mean(reward_tracker.episode_rewards[-50:]) if len(reward_tracker.episode_rewards) >= 50 else np.mean(reward_tracker.episode_rewards)
    final_success_rate = reward_tracker.success_rate[-1] if reward_tracker.success_rate else 0
    
    print(f"最终统计:")
    print(f"  - 总回合数: {len(reward_tracker.episode_rewards)}")
    print(f"  - 总步数: {callback.num_timesteps}")
    print(f"  - 最终平均奖励: {final_avg_reward:.2f}")
    print(f"  - 最终成功率: {final_success_rate:.2%}")
    print(f"  - 训练时长: {training_time/60:.1f}分钟")
    
    # 保存最终模型和数据
    final_model_path = PathConfig.FINAL_MODEL
    model.save(final_model_path)
    print(f"✅ 最终模型已保存: {final_model_path}")
    
    reward_tracker.plot_training_progress(PathConfig.FINAL_PROGRESS_PLOT)
    reward_tracker.save_data(PathConfig.FINAL_DATA_JSON)
    
    
    print("="*80)
    print(f"所有结果已保存:")
    print(f"  📊 日志和图表: agent/log_SB3/")
    print(f"  📁 模型文件: agent/model_SB3/")
    print("="*80)
    
    env.close()
    return final_model_path, reward_tracker, model


if __name__ == '__main__':
    # 运行继续训练
    model_path, reward_tracker, model = continue_train_leader_phase1_sb3(
        target_total_episodes=100000,  # 目标总回合数
        total_timesteps=None,          # 自动根据剩余回合数估算
        plot_interval=100,            # 每100回合绘制一次图
        load_model_path=None          # 自动查找最新模型
    )
    
    print(f"\n继续训练完成！检查 {PathConfig.LOG_DIR} 目录查看结果图表和数据。")