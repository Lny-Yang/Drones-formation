"""
第一阶段训练脚本 - 领航者单机导航训练

训练目标:
- 训练领航者无人机进行单机导航
- 学习避障和目标到达能力
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

# 导入进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("⚠️  tqdm 未安装，无法显示进度条: pip install tqdm")
    TQDM_AVAILABLE = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
        self.moving_avg_collision = []  # 新增滑动平均碰撞率
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
        # 计算奖励滑动平均
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
        # 碰撞率（仅用于滑动平均）
        self.collision_rate.append(1.0 if collision else 0.0)
        # 计算滑动平均碰撞率
        if len(self.collision_rate) >= self.window_size:
            avg_collision = np.mean(self.collision_rate[-self.window_size:])
        else:
            avg_collision = np.mean(self.collision_rate)
        self.moving_avg_collision.append(avg_collision)
        
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
        # 3. 碰撞率（仅滑动平均）
        ax3.plot(episodes, self.moving_avg_collision, color='orange', linewidth=2, label=f'{self.window_size}回合滑动平均')
        ax3.set_xlabel('回合数')
        ax3.set_ylabel('碰撞率')
        ax3.set_title(f'碰撞率变化（{self.window_size}回合滑动平均）')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        ax3.legend()
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
            'moving_avg_collision': [float(x) for x in self.moving_avg_collision],  # 🔥 添加这个字段
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
    
    def __init__(self, reward_tracker, max_episodes, plot_interval=500, save_interval=500, verbose=1, initial_episode=0):
        super(TrainingCallback, self).__init__(verbose)
        self.reward_tracker = reward_tracker
        self.max_episodes = max_episodes  # 🔥 添加最大回合数限制
        self.plot_interval = plot_interval
        self.save_interval = save_interval
        self.episode_count = initial_episode  # 🔥 支持从指定回合数开始
        self.episode_reward = 0
        self.episode_length = 0
        self.start_time = time.time()
        
        # 初始化进度条
        if TQDM_AVAILABLE:
            self.pbar = tqdm(total=max_episodes, initial=initial_episode, desc="训练进度", unit="回合", 
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        else:
            self.pbar = None
        
        
    def _on_step(self) -> bool:
        """每步调用"""
        try:
            # 🔥 检测NaN值（兼容numpy和tensor类型）
            reward = self.locals['rewards'][0]
            if isinstance(reward, np.ndarray):
                reward = reward.item()
            if np.isnan(reward) or np.isinf(reward):
                print(f"⚠️  警告: 在步骤 {self.num_timesteps} 检测到异常奖励值!")
                return False
            
            # 累积奖励
            self.episode_reward += self.locals['rewards'][0]
            self.episode_length += 1
            
            # 检查是否回合结束
            if self.locals['dones'][0]:
                self.episode_count += 1
                # 使用max_episodes作为总回合数，提前定义，避免UnboundLocalError
                total_episodes = self.max_episodes
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
                # 直接使用RewardTracker已计算好的滑动平均/窗口统计
                current_avg_reward = self.reward_tracker.moving_avg_rewards[-1] if self.reward_tracker.moving_avg_rewards else self.episode_reward
                current_success_rate = self.reward_tracker.success_rate[-1] if self.reward_tracker.success_rate else 0
                current_collision_rate = self.reward_tracker.moving_avg_collision[-1] if self.reward_tracker.moving_avg_collision else (1.0 if collision else 0.0)
                # 计算ETA
                elapsed_time = time.time() - self.start_time

                # 写入TensorBoard日志（奖励和成功率）
                try:
                    if hasattr(self.model, 'logger'):
                        self.model.logger.record('custom/episode_reward', self.episode_reward)
                        self.model.logger.record('custom/success_rate', current_success_rate)
                except Exception as e:
                    print(f"logger.record 失败: {e}")
                # 写入滑动平均碰撞率到TensorBoard
                try:
                    if hasattr(self.model, 'logger') and hasattr(self.reward_tracker, 'moving_avg_collision'):
                        self.model.logger.record('custom/avg_collision_rate', self.reward_tracker.moving_avg_collision[-1])
                except Exception as e:
                    print(f"logger.record collision 失败: {e}")

                eta = (elapsed_time / self.episode_count) * (total_episodes - self.episode_count) if self.episode_count > 0 else 0
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
                
                # 更新进度条
                if self.pbar:
                    self.pbar.update(1)
                    self.pbar.set_postfix({
                        '奖励': f"{self.episode_reward:7.2f}",
                        '平均': f"{current_avg_reward:6.2f}",
                        '成功率': f"{current_success_rate:5.1%}",
                        '碰撞率': f"{current_collision_rate:5.1%}",
                        'ETA': f"{eta/60:5.1f}分"
                    })
                
                # 每100回合打印一次详细进度信息（减少输出）
                if self.episode_count % 100 == 0:
                    print(f"回合 {self.episode_count:4d}/{total_episodes} | "
                        f"奖励: {self.episode_reward:8.2f} | "
                        f"平均: {current_avg_reward:6.2f} | "
                        f"成功率: {current_success_rate:.1%} | "
                        f"碰撞率: {current_collision_rate:.1%} | "
                        f"ETA: {eta/60:.1f}分钟")
                
                
                # 定期绘制和保存
                if self.episode_count % self.plot_interval == 0:
                    self.reward_tracker.plot_training_progress()
                    self.reward_tracker.save_data()
                    
                
                # 定期保存模型
                if self.episode_count % self.save_interval == 0:
                    # 🔥 保存前检查并裁剪log_std（防止保存爆炸的值）
                    log_std_val = self.model.policy.log_std.data
                    log_std_mean = log_std_val.mean().item()
                    log_std_max = log_std_val.max().item()
                    
                    # # 🎯 严格的阈值：[-0.8, 0.0]（std范围: 0.45~1.0）
                    # # 理由：从头训练时，应该保持较低的探索噪声，避免log_std失控
                    # if log_std_max > 0.0 or log_std_mean > -0.4:
                    #     print(f"⚠️  检测到log_std增长: 均值={log_std_mean:.4f}, 最大={log_std_max:.4f}")
                    #     print(f"    对应std: 均值={np.exp(log_std_mean):.2f}, 最大={np.exp(log_std_max):.2f}")
                    #     print(f"    → 裁剪到 [-0.8, 0.0] (std范围: 0.45~1.0)")
                    #     self.model.policy.log_std.data.clamp_(-0.8, 0.0)
                    #     new_mean = self.model.policy.log_std.data.mean().item()
                    #     print(f"    ✅ 裁剪后均值={new_mean:.4f} (std≈{np.exp(new_mean):.2f})")
                    
                    model_path = PathConfig.get_episode_model_path(self.episode_count)
                    self.model.save(model_path)
                    print(f"✅ 模型已保存: {model_path}")
                    print("-" * 80)
                
                # 重置回合统计
                self.episode_reward = 0
                self.episode_length = 0
                
                # 🔥 检查是否达到最大回合数，如果是则停止训练
                if self.episode_count >= self.max_episodes:
                    print("="*80)
                    print(f"✅ 已完成 {self.max_episodes} 回合训练，停止训练")
                    print("="*80)
                    return False  # 返回False停止训练
            
            return True
        except Exception as e:
            print(f"_on_step 异常: {e}")
            import traceback
            traceback.print_exc()
            return False

def make_env(max_steps=1000):  # 🔥 增加到1000步，提供更多探索时间
    """创建环境的工厂函数
    
    Args:
        max_steps: 每个回合的最大步数 (默认1000)
    """

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


def train_leader_phase1_sb3(max_episodes=20000, total_timesteps=None, plot_interval=500, 
                           continue_training=False, load_model_path=None):
    """第一阶段训练 - 领航者单机导航训练（使用PPO算法）
    
    Args:
        max_episodes: 最大训练回合数 (默认20000)
        total_timesteps: 总训练步数（如果为None，则根据max_episodes估算）
        plot_interval: 绘图和保存间隔
        continue_training: 是否继续训练（加载之前的模型）
        load_model_path: 要加载的模型路径（如果为None且continue_training=True，则加载最新模型）
    """
    print("="*80)
    if continue_training:
        print("第一阶段训练 - 领航者导航训练（继续训练）")
    else:
        print("第一阶段训练 - 领航者导航训练（从头开始）")
    print("="*80)
    
    # 确保所有目录存在
    PathConfig.ensure_directories()
    
    # 创建单环境
    print("正在创建环境...")
    env = DummyVecEnv([make_env(max_steps=1000)])  # 🔥 1000步/回合，提供更多探索时间
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
    # 设置环境的总回合数（用于ETA计算）
    test_env.max_episodes = max_episodes
    
    # 估算总步数（如果未指定）
    if total_timesteps is None:
        # 🔥 使用更保守的估计，确保不会因为步数限制而过早停止
        # 假设平均每回合600步（max_steps=1000，考虑到探索和失败的情况）
        avg_steps_per_episode = 600
        total_timesteps = max_episodes * avg_steps_per_episode
        print(f"  - 估算总步数: {total_timesteps:,} ({max_episodes}回合 × {avg_steps_per_episode}步)")
        print(f"  ⚠️  注意: 实际训练将在达到 {max_episodes} 回合时停止（由回调函数控制）")
    
    print("="*80)
    
    # 🔥 MLP策略强制使用CPU（比GPU快3-5倍！）
    # 参考：https://github.com/DLR-RM/stable-baselines3/issues/1245
    device_name = 'cpu'
    print(f"✅ 使用CPU训练 (MLP策略在CPU上比GPU更快3-5倍)")
    if torch.cuda.is_available():
        print(f"   检测到GPU: {torch.cuda.get_device_name(0)} (但MLP不适合GPU加速)")
    
    # 🔄 检查是否继续训练
    if continue_training:
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
                    print(f"� 找到最新episode模型，加载: {load_model_path}")
                else:
                    print("⚠️  未找到任何已保存模型，将从头开始训练")
                    continue_training = False
        else:
            print(f"🔄 加载指定模型: {load_model_path}")
            # 如果是字符串，转换为完整的路径
            if isinstance(load_model_path, str):
                load_model_path = PathConfig.get_episode_model_path(int(load_model_path.split('_')[-1]))
        
        if continue_training:
            try:
                print("正在加载模型...")
                model = PPO.load(load_model_path, env=env, device=device_name)
                print("✅ 模型加载成功！")
                
                # 尝试加载训练历史数据
                if PathConfig.TRAINING_DATA_JSON.exists():
                    print("\n正在加载训练历史...")
                    with open(PathConfig.TRAINING_DATA_JSON, 'r', encoding='utf-8') as f:
                        history_data = json.load(f)
                    print(f"✅ 已加载 {history_data['total_episodes']} 回合的历史数据")
                else:
                    print("⚠️  未找到训练历史数据，将重新统计")
                    history_data = None
            except Exception as e:
                print(f"❌ 加载模型失败: {e}")
                print("将从头开始训练")
                continue_training = False
                history_data = None
    
    # 🆕 创建或已加载模型
    if not continue_training:
        print("创建新的PPO模型...")
        
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,        # 学习率（标准值）
            n_steps=2048,               # 🔥 降低到512（从2048），更频繁更新，加快迭代速度
            batch_size=64,             # Mini-batch大小
            n_epochs=10,                # 🔥 减少到4（从10），加快单次更新速度
            gamma=0.99,                # 折扣因子
            gae_lambda=0.95,           # GAE参数
            clip_range=0.2,            # PPO裁剪范围
            clip_range_vf=0.2,         # Value function裁剪，稳定价值估计
            ent_coef=0.01,             # 🔥 降低熵系数（0.03→0.01），减少探索推动力
            vf_coef=0.5,               # Value function损失系数
            max_grad_norm=0.5,         # 梯度裁剪阈值（防止梯度爆炸）
            use_sde=False,             # 不使用状态依赖探索
            sde_sample_freq=-1,
            target_kl=None,            # 不使用KL散度early stopping，依赖clip机制
            tensorboard_log=str(PathConfig.LOG_DIR / "tensorboard"),     
            policy_kwargs=dict(
                # 🎯 网络结构设计（针对140维观测空间 → 2维动作）:
                # 输入140维 → Actor[256→128] → 动作2维
                # 输入140维 → Critic[256→128] → 价值1维
                net_arch=[dict(pi=[256, 128], vf=[256, 128])],
                activation_fn=torch.nn.Tanh,
                ortho_init=True,
                log_std_init=-0.8,  # 🔥 初始化log_std为-0.8 (std≈0.45)，更保守的探索
            ),
            verbose=1,
            seed=SEED,                 # 🔥 设置随机种子，保证可复现
            device=device_name,
        )
    
    print("模型配置:")
    # 🔥 处理schedule函数显示
    lr_val = model.learning_rate(1.0) if callable(model.learning_rate) else model.learning_rate
    clip_val = model.clip_range(1.0) if callable(model.clip_range) else model.clip_range
    clip_vf_val = model.clip_range_vf(1.0) if callable(model.clip_range_vf) else model.clip_range_vf
    
    print(f"  - 学习率: {lr_val}")
    print(f"  - Batch大小: {model.batch_size}")
    print(f"  - 训练轮数: {model.n_epochs}")
    print(f"  - N_steps: {model.n_steps}")
    print(f"  - Gamma: {model.gamma}")
    print(f"  - GAE Lambda: {model.gae_lambda}")
    print(f"  - Clip范围: {clip_val} (PPO核心机制)")
    print(f"  - Clip范围(VF): {clip_vf_val} (稳定价值估计)")
    print(f"  - 熵系数: {model.ent_coef}")
    print(f"  - VF系数: {model.vf_coef}")
    print(f"  - 梯度裁剪: {model.max_grad_norm} (防止梯度爆炸)")
    print(f"  - Target KL: {model.target_kl} (标准PPO)")
    
    # 🔥 显示当前log_std值
    current_log_std = model.policy.log_std.data.cpu().numpy()
    print(f"  - Log_std: 均值={current_log_std.mean():.4f}, "
          f"范围=[{current_log_std.min():.4f}, {current_log_std.max():.4f}], "
          f"对应std≈{np.exp(current_log_std.mean()):.4f}")
    print(f"  - 设备: {model.device}")
    print("="*80)
    
    # 创建奖励跟踪器
    reward_tracker = RewardTracker(window_size=500)  # 🔥 使用500回合窗口，减少统计噪声
    
    # 🔄 如果继续训练，恢复历史数据
    # 🔥 先确定要恢复到哪个episode（优先使用模型文件名的episode数）
    target_episode = 0
    if continue_training and 'load_model_path' in locals() and load_model_path:
        if isinstance(load_model_path, (str, Path)):
            model_name = Path(load_model_path).name
            if 'episode_' in model_name:
                try:
                    target_episode = int(model_name.split('episode_')[-1])
                    print(f"🎯 目标恢复到episode: {target_episode}")
                except:
                    pass
    
    if continue_training and 'history_data' in locals() and history_data:
        print("正在恢复训练历史...")
        
        # 🔥 加载所有历史数据
        all_rewards = history_data.get('episode_rewards', [])
        all_lengths = history_data.get('episode_lengths', [])
        all_moving_avg = history_data.get('moving_avg_rewards', [])
        all_success_rate = history_data.get('success_rate', [])
        all_success_flags = history_data.get('success_flags', [])
        all_collision_rate = history_data.get('collision_rate', [])
        all_moving_avg_collision = history_data.get('moving_avg_collision', [])
        
        # 🔥 如果目标episode < 历史数据总数，截断到目标episode
        if target_episode > 0 and target_episode < len(all_rewards):
            print(f"  ⚠️  历史数据有 {len(all_rewards)} 回合，但模型是episode_{target_episode}")
            print(f"  🔪 截断历史数据到前 {target_episode} 回合（丢弃后续数据）")
            
            reward_tracker.episode_rewards = all_rewards[:target_episode]
            reward_tracker.episode_lengths = all_lengths[:target_episode]
            reward_tracker.moving_avg_rewards = all_moving_avg[:target_episode]
            reward_tracker.success_rate = all_success_rate[:target_episode]
            reward_tracker.success_flags = all_success_flags[:target_episode]
            reward_tracker.collision_rate = all_collision_rate[:target_episode]
            
            # 对于moving_avg_collision，如果长度不匹配则重新计算
            if len(all_moving_avg_collision) >= target_episode:
                reward_tracker.moving_avg_collision = all_moving_avg_collision[:target_episode]
            else:
                print("  ⚠️  moving_avg_collision长度不足，重新计算...")
                reward_tracker.moving_avg_collision = []
                for i in range(target_episode):
                    if i >= reward_tracker.window_size:
                        avg_collision = np.mean(reward_tracker.collision_rate[i-reward_tracker.window_size+1:i+1])
                    else:
                        avg_collision = np.mean(reward_tracker.collision_rate[:i+1])
                    reward_tracker.moving_avg_collision.append(avg_collision)
        else:
            # 正常恢复所有数据
            reward_tracker.episode_rewards = all_rewards
            reward_tracker.episode_lengths = all_lengths
            reward_tracker.moving_avg_rewards = all_moving_avg
            reward_tracker.success_rate = all_success_rate
            reward_tracker.success_flags = all_success_flags
            reward_tracker.collision_rate = all_collision_rate
            
            # 重新计算moving_avg_collision（如果历史数据中没有）
            if all_moving_avg_collision:
                reward_tracker.moving_avg_collision = all_moving_avg_collision
            else:
                print("  ⚠️  历史数据缺少moving_avg_collision，重新计算...")
                reward_tracker.moving_avg_collision = []
                for i in range(len(reward_tracker.collision_rate)):
                    if i >= reward_tracker.window_size:
                        avg_collision = np.mean(reward_tracker.collision_rate[i-reward_tracker.window_size+1:i+1])
                    else:
                        avg_collision = np.mean(reward_tracker.collision_rate[:i+1])
                    reward_tracker.moving_avg_collision.append(avg_collision)
        
        print(f"✅ 已恢复 {len(reward_tracker.episode_rewards)} 回合的训练历史")
        if reward_tracker.success_rate:
            print(f"   恢复点成功率: {reward_tracker.success_rate[-1]:.1%}")
        if reward_tracker.moving_avg_rewards:
            print(f"   恢复点平均奖励: {reward_tracker.moving_avg_rewards[-1]:.2f}")
        if reward_tracker.moving_avg_collision:
            print(f"   恢复点碰撞率: {reward_tracker.moving_avg_collision[-1]:.1%}")
    
    # 创建回调函数
    initial_episode = 0
    if continue_training and 'history_data' in locals() and history_data:
        # 🔥 从模型文件名提取episode数，而不是从历史数据
        if isinstance(load_model_path, (str, Path)):
            model_name = Path(load_model_path).name
            if 'episode_' in model_name:
                try:
                    # 提取episode数字（例如：leader_phase1_episode_99000 → 99000）
                    episode_num = int(model_name.split('episode_')[-1])
                    initial_episode = episode_num
                    print(f"🔄 从模型文件episode数继续训练: {initial_episode}")
                except:
                    # 如果提取失败，使用历史数据
                    initial_episode = history_data.get('total_episodes', 0)
                    print(f"🔄 从历史数据继续训练: {initial_episode}")
            else:
                initial_episode = history_data.get('total_episodes', 0)
                print(f"🔄 从历史数据继续训练: {initial_episode}")
        else:
            initial_episode = history_data.get('total_episodes', 0)
            print(f"🔄 从历史数据继续训练: {initial_episode}")
    
    callback = TrainingCallback(
        reward_tracker=reward_tracker,
        max_episodes=max_episodes,  # 🔥 传入最大回合数
        plot_interval=plot_interval,
        save_interval=plot_interval,
        initial_episode=initial_episode  # 🔥 传入初始回合数
    )
    
    # 开始训练
    print("开始训练...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=5000,  # 每5000个episode打印一次日志，减少print
            progress_bar=False  # 我们使用自定义进度显示
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断！")
    
    # 训练完成
    training_time = time.time() - start_time
    print("="*80)
    print("训练完成！")
    
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
    
    # 关闭进度条
    if hasattr(callback, 'pbar') and callback.pbar:
        callback.pbar.close()
    
    env.close()
    return final_model_path, reward_tracker, model


if __name__ == '__main__':
    # 🔥 从头开始训练 - 优化配置
    # 关键改进：
    # 1. ent_coef: 0.03 → 0.01 (降低熵推动，减缓log_std增长)
    # 2. log_std_init: -1.0 → -0.8 (更保守的初始探索，std≈0.45)
    # 3. n_steps: 1024 → 2048 (收集更多经验再更新，提高稳定性)
    # 4. log_std裁剪: [-0.5,0.3] → [-0.8,0.0] (更严格的控制)
    # 5. seed=SEED (保证可复现)
    
    model_path, reward_tracker, model = train_leader_phase1_sb3(
        max_episodes=100000,       # 训练回合数
        total_timesteps=None,      # 自动根据回合数估算
        plot_interval=1000,        # 每1000回合保存一次检查点
        continue_training=True,    # 🔥 从头开始训练
        load_model_path='leader_phase1_episode_14000'      # 从14000回合继续训练
    )
    
    print(f"\n训练完成！检查 {PathConfig.LOG_DIR} 目录查看结果图表和数据。")

