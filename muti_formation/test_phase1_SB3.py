"""
第一阶段模型测试脚本（Stable-Baselines3版本）
测试领航者避障和导航性能
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import argparse
import random
import torch
from collections import defaultdict

# 导入stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # 🔥 添加：与训练一致

from drone_envs.envs.drone_env_multi import DroneNavigationMulti
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def make_env(num_drones=1, max_steps=1000):
    """创建环境的工厂函数 - 与训练脚本完全一致"""

    def _init():
        env = DroneNavigationMulti(
            num_drones=num_drones,
            use_depth_camera=True,
            depth_camera_range=10.0,
            depth_resolution=16,
            enable_formation_force=False,
            training_stage=1,
            max_steps=max_steps
        )
        return env
    return _init

def test_model(model_path, num_episodes=20, max_steps=1000, render=True):
    """测试第一阶段训练的SB3 PPO模型"""
    print("="*80)
    print("第一阶段模型测试（Stable-Baselines3 PPO）：领航者避障和导航")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"测试回合数: {num_episodes}")
    print(f"最大步数: {max_steps}")
    
    # 🔥 创建环境 - 与训练环境完全一致（使用DummyVecEnv包装）
    env = DummyVecEnv([make_env(num_drones=1, max_steps=max_steps)])
    
    # 获取底层环境用于检查配置
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
    print(f"  - 环境包装: DummyVecEnv (与训练一致)")  # 🔥 新增
    
    # 加载SB3 PPO模型
    if os.path.exists(model_path + '.zip') or os.path.exists(model_path):
        try:
            # SB3会自动添加.zip后缀
            model = PPO.load(model_path, env=env)
            print(f"✓ 模型加载成功: {model_path}")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            return None
    else:
        print(f"✗ 模型文件不存在: {model_path}")
        return None
    
    print("="*80)
    
    # 测试统计
    test_results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_count': 0,
        'collision_count': 0,
        'boundary_collision_count': 0,
        'physical_collision_count': 0,
        'timeout_count': 0,
        'min_depths': [],
        'goal_distances': [],
        'reward_components': defaultdict(list)  # 记录各个奖励分量
    }
    
    # 开始测试
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()  # 🔥 DummyVecEnv返回的是(1, obs_dim)
        episode_reward = 0
        obstacle_detections = 0
        collision_occurred = False
        collision_type = ""
        min_depths = []
        episode_reward_components = defaultdict(float)
        
        for step in range(max_steps):
            # 🔥 获取底层环境进行监控
            test_env = env.envs[0]
            
            # 监控避障信息
            if hasattr(test_env, 'depth_obstacle_processor') and test_env.use_leader_camera:
                try:
                    # 使用屏蔽后的深度图像进行避障检测，避免无人机自身被误认为障碍物
                    depth_image = test_env._get_masked_leader_depth()
                    if depth_image is not None and depth_image.size > 0:
                        raw_depth = depth_image if len(depth_image.shape) == 2 else depth_image[:, :, 0]
                        processed_depth = test_env.depth_obstacle_processor.preprocess_depth_image(raw_depth)
                        obstacle_detected, min_depth = test_env.depth_obstacle_processor.detect_obstacles(processed_depth)
                        
                        min_depths.append(min_depth)
                        if obstacle_detected:
                            obstacle_detections += 1
                except Exception:
                    pass
            
            # 使用SB3模型预测动作（确定性策略，无探索噪声）
            action, _states = model.predict(state, deterministic=True)
            
            # 🔥 环境步进 - DummyVecEnv返回的都是数组形式
            next_state, reward, done, info = env.step(action)
            episode_reward += reward[0]  # 🔥 reward是数组，取第一个元素
            
            # 🔥 info也是列表形式
            info = info[0]
            
            # 记录奖励分量
            reward_info = info.get('reward_info', {})
            for key, value in reward_info.items():
                episode_reward_components[key] += value
            
            # 检查碰撞类型
            crash_reward = reward_info.get('crash', 0)
            if crash_reward < 0:
                collision_occurred = True
                if hasattr(test_env, '_get_depth_info'):
                    try:
                        depth_info = env._get_depth_info()
                        collision_type = depth_info.get('collision_type', 'unknown')
                    except:
                        collision_type = "碰撞"
            
            # 渲染（如果启用）
            if render:
                env.render()
                time.sleep(0.01)  # 控制渲染速度
            
            state = next_state
            
            # 🔥 DummyVecEnv的done是数组
            if done[0]:
                break
        
        # 统计结果
        success = info.get('success', False)
        
        test_results['episode_rewards'].append(episode_reward)
        test_results['episode_lengths'].append(step + 1)
        if min_depths:
            test_results['min_depths'].append(np.mean(min_depths))
        
        # 记录奖励分量
        for key, value in episode_reward_components.items():
            test_results['reward_components'][key].append(value)
        
        # 记录结果类型
        if success:
            test_results['success_count'] += 1
            result_str = "✓ 成功"
        elif collision_occurred:
            test_results['collision_count'] += 1
            if collision_type == 'boundary':
                test_results['boundary_collision_count'] += 1
                result_str = "✗ 边界碰撞"
            else:
                test_results['physical_collision_count'] += 1
                result_str = f"✗ {collision_type}"
        elif step + 1 >= max_steps:
            test_results['timeout_count'] += 1
            result_str = "⏱ 超时"
        else:
            result_str = "? 其他"
        
        # 计算到目标距离
        if hasattr(test_env, 'goal') and test_env.goal is not None:
            leader_pos, _ = test_env.drones[0].get_position_and_orientation() if hasattr(test_env.drones[0], 'get_position_and_orientation') else ([0,0,0], [0,0,0,1])
            goal_distance = np.linalg.norm(np.array(leader_pos) - np.array(test_env.goal))
            test_results['goal_distances'].append(goal_distance)
        
        print(f"回合 {episode + 1:2d}/{num_episodes} | "
              f"奖励: {episode_reward:7.2f} | "
              f"步数: {step + 1:4d} | "
              f"障碍物检测: {obstacle_detections:3d} | "
              f"结果: {result_str}")
    
    # 计算最终统计
    total_time = time.time() - start_time
    success_rate = test_results['success_count'] / num_episodes
    collision_rate = test_results['collision_count'] / num_episodes
    avg_reward = np.mean(test_results['episode_rewards'])
    avg_length = np.mean(test_results['episode_lengths'])
    
    print("="*80)
    print("测试结果统计:")
    print("="*80)
    print(f"总回合数: {num_episodes}")
    print(f"成功回合: {test_results['success_count']} ({success_rate:.1%})")
    print(f"碰撞回合: {test_results['collision_count']} ({collision_rate:.1%})")
    print(f"  - 边界碰撞: {test_results['boundary_collision_count']}")
    print(f"  - 物理碰撞: {test_results['physical_collision_count']}")
    print(f"超时回合: {test_results['timeout_count']} ({test_results['timeout_count']/num_episodes:.1%})")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_length:.1f}")
    if test_results['min_depths']:
        print(f"平均最小深度: {np.mean(test_results['min_depths']):.2f}m")
    if test_results['goal_distances']:
        print(f"平均目标距离: {np.mean(test_results['goal_distances']):.2f}m")
    
    # 打印奖励分量统计
    if test_results['reward_components']:
        print(f"\n奖励分量平均值:")
        for key in sorted(test_results['reward_components'].keys()):
            values = test_results['reward_components'][key]
            avg_value = np.mean(values)
            print(f"  - {key}: {avg_value:.2f}")
    
    print(f"\n测试时长: {total_time:.1f}秒")
    print("="*80)
    
    env.close()
    
    # 转换defaultdict为普通dict以便JSON序列化
    test_results['reward_components'] = {k: list(v) for k, v in test_results['reward_components'].items()}
    
    return test_results

def plot_test_results(results, model_name, save_dir):
    """绘制测试结果图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'第一阶段模型测试结果（SB3 PPO）- {model_name}', fontsize=16)
    
    episodes = range(1, len(results['episode_rewards']) + 1)
    
    # 1. 奖励曲线
    ax1.plot(episodes, results['episode_rewards'], 'b-', alpha=0.7, marker='o', markersize=3)
    ax1.axhline(y=np.mean(results['episode_rewards']), color='r', linestyle='--', 
                label=f'平均值: {np.mean(results["episode_rewards"]):.1f}')
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('奖励')
    ax1.set_title('回合奖励')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 回合长度
    ax2.plot(episodes, results['episode_lengths'], 'g-', alpha=0.7, marker='s', markersize=3)
    ax2.axhline(y=np.mean(results['episode_lengths']), color='r', linestyle='--', 
                label=f'平均值: {np.mean(results["episode_lengths"]):.1f}')
    ax2.set_xlabel('回合数')
    ax2.set_ylabel('步数')
    ax2.set_title('回合长度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 结果类型统计
    result_types = ['成功', '边界碰撞', '物理碰撞', '超时']
    result_counts = [results['success_count'], 
                    results['boundary_collision_count'],
                    results['physical_collision_count'],
                    results['timeout_count']]
    
    colors = ['lightgreen', 'orange', 'red', 'gray']
    bars = ax3.bar(result_types, result_counts, color=colors)
    ax3.set_ylabel('回合数')
    ax3.set_title('测试结果类型统计')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    # 在柱状图上标注数值
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
    
    # 4. 结果统计饼图
    total = len(results['episode_rewards'])
    labels = [f'{label}\n({count}/{total})' 
              for label, count in zip(result_types, result_counts)]
    
    # 只显示非零项
    non_zero_sizes = [s for s in result_counts if s > 0]
    non_zero_labels = [l for l, s in zip(labels, result_counts) if s > 0]
    non_zero_colors = [c for c, s in zip(colors, result_counts) if s > 0]
    
    if non_zero_sizes:
        ax4.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, 
                autopct='%1.1f%%', startangle=90)
        ax4.set_title('测试结果分布')
    else:
        ax4.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = save_dir / f"phase1_test_results_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 测试结果图表已保存: {save_path}")
    
    plt.show()
    return fig

def main():
    parser = argparse.ArgumentParser(description='测试第一阶段训练的SB3 PPO模型')
    parser.add_argument('--model', type=str, 
                       default='muti_formation/agent/model_SB3/leader_phase1_episode_60000',
                       help='模型文件路径（不含.zip后缀）')
    parser.add_argument('--episodes', type=int, default=50,
                       help='测试回合数')
    parser.add_argument('--max_steps', type=int, default=1000,  # 与训练一致
                       help='每回合最大步数')
    parser.add_argument('--no_render', action='store_true',
                       help='禁用渲染（加快测试速度）')
    parser.add_argument('--save_dir', type=str, default='muti_formation/agent/log_SB3',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 确保保存目录存在
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试模型
    results = test_model(
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=not args.no_render
    )
    
    if results is not None:
        # 保存测试结果
        model_name = Path(args.model).stem
        results_path = save_dir / f"phase1_test_results_{model_name}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 测试结果数据已保存: {results_path}")
        
        # 绘制结果图表
        plot_test_results(results, model_name, save_dir)

if __name__ == '__main__':
    main()
