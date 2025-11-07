"""
第一阶段模型测试脚本：测试领航者避障和导航性能
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import argparse
from collections import defaultdict

from agent.CTDE_PPO_agent import CTDE_PPO, device
from drone_envs.envs.drone_env_multi import DroneNavigationMulti

def test_model(model_path, num_episodes=20, max_steps=1500, render=True):
    """测试第一阶段训练的模型"""
    print("="*80)
    print("第一阶段模型测试：领航者避障和导航")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"测试回合数: {num_episodes}")
    print(f"最大步数: {max_steps}")
    
    # 创建环境 - 与训练环境完全一致
    env = DroneNavigationMulti(
        num_drones=1,  # 与训练一致：只测试领航者
        use_depth_camera=True,
        depth_camera_range=10.0,
        depth_resolution=16,
        enable_formation_force=False,  # 关闭编队力
        training_stage=1,  # 第一阶段
        max_steps=max_steps  # 使用传入的最大步数
    )
    
    # 创建CTDE代理 - 与训练环境完全一致
    # 平面模式：位置(2) + 速度(2) + 朝向(4) + 目标相对位置(2) + 深度特征(130) = 140维
    leader_state_dim = 2 + 2 + 4 + 2 + env.depth_feature_dim if env.enforce_planar else 3 + 3 + 4 + 3 + env.depth_feature_dim
    
    ppo_agent = CTDE_PPO(
        leader_state_dim=leader_state_dim,
        follower_state_dim=leader_state_dim,  # 跟随者状态维度与领航者相同
        leader_visual_dim=env.depth_feature_dim,
        action_dim=2,  # 前进/后退力和转向扭矩
        num_drones=1,  # 第一阶段只训练领航者
        lr_actor=0.0003,    # 学习率保持
        lr_critic=0.001,    # 学习率保持
        gamma=0.99,         # 折扣因子
        K_epochs=40,        # 更新轮数，避免过拟合
        eps_clip=0.2,       # PPO裁剪参数
        has_continuous_action_space=True,
        action_std_init=0.15  # 与训练一致：降低探索噪声
    )
    
    # 加载模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # 加载领航者策略
        if 'leader_policy' in checkpoint:
            ppo_agent.leader_policy.load_state_dict(checkpoint['leader_policy'])
            ppo_agent.leader_policy_old.load_state_dict(checkpoint['leader_policy'])
            print(f"✓ 领航者策略加载成功")
        
        # 加载跟随者策略（如果有）
        if 'follower_policies' in checkpoint and len(checkpoint['follower_policies']) > 0:
            for i, policy_state in enumerate(checkpoint['follower_policies']):
                if i < len(ppo_agent.follower_policies):
                    ppo_agent.follower_policies[i].load_state_dict(policy_state)
                    ppo_agent.follower_policies_old[i].load_state_dict(policy_state)
            print(f"✓ 跟随者策略加载成功 ({len(checkpoint['follower_policies'])} 个)")
        
        # 加载全局评论家（如果有）
        if 'global_critic' in checkpoint and ppo_agent.global_critic is not None:
            ppo_agent.global_critic.load_state_dict(checkpoint['global_critic'])
            print(f"✓ 全局评论家加载成功")
        
        print(f"✓ 模型加载完成: {model_path}")
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
        'goal_distances': []
    }
    
    # 开始测试
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        obstacle_detections = 0
        collision_occurred = False
        collision_type = ""
        min_depths = []
        
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
                        
                        min_depths.append(min_depth)
                        if obstacle_detected:
                            obstacle_detections += 1
                except Exception:
                    pass
            
            # 第一阶段：只控制领航者（使用训练好的策略，无探索噪声）
            with torch.no_grad():
                leader_action = ppo_agent.select_action([leader_obs], depth_features)[0]
            
            # 环境步进 - 第一阶段直接使用领航者动作
            next_state, reward, terminated, truncated, info = env.step(leader_action)
            episode_reward += reward
            
            # 检查碰撞类型
            reward_info = info.get('reward_info', {})
            crash_reward = reward_info.get('crash', 0)
            if crash_reward < 0:
                collision_occurred = True
                if hasattr(env, '_get_depth_info'):
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
            
            if terminated or truncated:
                break
        
        # 统计结果
        success = info.get('success', False)
        
        test_results['episode_rewards'].append(episode_reward)
        test_results['episode_lengths'].append(step + 1)
        if min_depths:
            test_results['min_depths'].append(np.mean(min_depths))
        
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
        if hasattr(env, 'goal') and env.goal is not None:
            leader_pos, _ = env.drones[0].get_position_and_orientation() if hasattr(env.drones[0], 'get_position_and_orientation') else ([0,0,0], [0,0,0,1])
            goal_distance = np.linalg.norm(np.array(leader_pos) - np.array(env.goal))
            test_results['goal_distances'].append(goal_distance)
        
        print(f"回合 {episode + 1:2d}/{num_episodes} | "
              f"奖励: {episode_reward:7.2f} | "
              f"步数: {step + 1:4d} | "
              f"障碍物检测: {obstacle_detections} | "
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
    print(f"测试时长: {total_time:.1f}秒")
    print("="*80)
    
    env.close()
    return test_results

def plot_test_results(results, model_name):
    """绘制测试结果图表"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'第一阶段模型测试结果 - {model_name}', fontsize=16)
    
    episodes = range(1, len(results['episode_rewards']) + 1)
    
    # 1. 奖励曲线
    ax1.plot(episodes, results['episode_rewards'], 'b-', alpha=0.7, marker='o', markersize=3)
    ax1.axhline(y=np.mean(results['episode_rewards']), color='r', linestyle='--', label=f'平均值: {np.mean(results["episode_rewards"]):.1f}')
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('奖励')
    ax1.set_title('回合奖励')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 回合长度
    ax2.plot(episodes, results['episode_lengths'], 'g-', alpha=0.7, marker='s', markersize=3)
    ax2.axhline(y=np.mean(results['episode_lengths']), color='r', linestyle='--', label=f'平均值: {np.mean(results["episode_lengths"]):.1f}')
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
    
    ax3.bar(result_types, result_counts, color=['lightgreen', 'orange', 'red', 'gray'])
    ax3.set_ylabel('回合数')
    ax3.set_title('测试结果类型统计')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 结果统计饼图
    total = len(results['episode_rewards'])
    labels = ['成功', '边界碰撞', '物理碰撞', '超时']
    sizes = [results['success_count'], 
             results['boundary_collision_count'],
             results['physical_collision_count'],
             results['timeout_count']]
    colors = ['lightgreen', 'orange', 'red', 'gray']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('测试结果分布')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = f"muti_formation/agent/log/phase1_test_results_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"测试结果图表已保存: {save_path}")
    
    plt.show()
    return fig

def main():
    parser = argparse.ArgumentParser(description='测试第一阶段训练的模型')
    parser.add_argument('--model', type=str, 
                       default='muti_formation/agent/model/leader_phase1_episode_15000.pth',
                       help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=100,
                       help='测试回合数')
    parser.add_argument('--max_steps', type=int, default=3000,  # 与训练一致
                       help='每回合最大步数')
    parser.add_argument('--no_render', action='store_true',
                       help='禁用渲染（加快测试速度）')
    
    args = parser.parse_args()
    
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
        results_path = f"muti_formation/agent/log/phase1_test_results_{model_name}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"测试结果数据已保存: {results_path}")
        
        # 绘制结果图表
        plot_test_results(results, model_name)

if __name__ == '__main__':
    main()