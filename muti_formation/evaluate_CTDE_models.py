
"""
分阶段CTDE模型评估脚本
评估领航者-跟随者编队控制系统的性能
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from agent.CTDE_PPO_agent import CTDE_PPO
from drone_envs.envs.drone_env_multi import DroneNavigationMulti
import json

def evaluate_model(env, ppo_agent, model_path, num_episodes=10, max_steps=300):
    """评估模型性能"""
    print(f"加载模型: {model_path}")
    ppo_agent.load(model_path)

    episode_rewards = []
    leader_rewards = []
    follower_rewards = []
    success_rates = []
    formation_errors = []
    obstacle_collisions = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        leader_reward_total = 0
        follower_reward_total = 0
        formation_error_total = 0
        collision_count = 0

        for step in range(max_steps):
            # 提取深度特征
            depth_features = state[-env.depth_feature_dim:] if env.use_depth_camera else None

            # 选择动作
            actions = ppo_agent.select_action([state] * env.num_drones, depth_features)
            combined_actions = np.concatenate(actions)

            next_state, reward, terminated, truncated, info = env.step(combined_actions)

            # 收集奖励信息 - 适应新的4奖励系统
            reward_info = info.get('reward_info', {})
            # 新的奖励系统是统一的，使用总奖励作为领航者奖励
            leader_reward = reward  # 总奖励即领航者奖励
            follower_reward = 0.0   # 跟随者奖励设为0（新系统不区分）

            leader_reward_total += leader_reward
            follower_reward_total += follower_reward
            episode_reward += reward

            # 收集编队误差 - 新系统不再提供formation_error
            formation_error = 0.0  # 设为0，新系统不计算编队误差
            formation_error_total += formation_error

            # 检测碰撞 - 使用新的crash奖励
            crash_reward = reward_info.get('crash', 0)
            if crash_reward < 0:
                collision_count += 1

            state = next_state

            if terminated or truncated:
                break

        # 计算成功率（到达目标且无碰撞）
        success = terminated and collision_count == 0
        success_rates.append(1 if success else 0)

        episode_rewards.append(episode_reward)
        leader_rewards.append(leader_reward_total)
        follower_rewards.append(follower_reward_total)
        formation_errors.append(formation_error_total / max_steps)
        obstacle_collisions.append(collision_count)

        print(f"评估回合 {episode + 1}/{num_episodes} | 奖励: {episode_reward:.2f} | 领航者: {leader_reward_total:.2f} | 跟随者: {follower_reward_total:.2f} | 成功: {success}")

    results = {
        'episode_rewards': episode_rewards,
        'leader_rewards': leader_rewards,
        'follower_rewards': follower_rewards,
        'success_rates': success_rates,
        'obstacle_collisions': obstacle_collisions,
        'avg_reward': np.mean(episode_rewards),
        'avg_leader_reward': np.mean(leader_rewards),
        'avg_follower_reward': np.mean(follower_rewards),
        'success_rate': np.mean(success_rates),
        'avg_collisions': np.mean(obstacle_collisions)
    }

    return results

def plot_evaluation_results(results, model_name):
    """绘制评估结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'模型评估结果 - {model_name} (4奖励系统)', fontsize=16)

    # 奖励分布
    axes[0, 0].hist(results['episode_rewards'], bins=10, alpha=0.7, color='blue')
    axes[0, 0].set_title('回合奖励分布')
    axes[0, 0].set_xlabel('奖励')
    axes[0, 0].set_ylabel('频次')

    # 领航者vs跟随者奖励
    axes[0, 1].scatter(results['leader_rewards'], results['follower_rewards'], alpha=0.7)
    axes[0, 1].set_title('领航者vs跟随者奖励')
    axes[0, 1].set_xlabel('领航者奖励')
    axes[0, 1].set_ylabel('跟随者奖励')

    # 编队误差 - 移除，新系统不计算编队误差
    axes[0, 2].text(0.5, 0.5, '编队误差\n(新系统已移除)', 
                   ha='center', va='center', fontsize=12, color='gray')
    axes[0, 2].set_title('编队误差 (已移除)')
    axes[0, 2].axis('off')

    # 碰撞统计
    axes[1, 0].bar(['无碰撞', '有碰撞'], [
        sum(1 for c in results['obstacle_collisions'] if c == 0),
        sum(1 for c in results['obstacle_collisions'] if c > 0)
    ], color=['green', 'red'])
    axes[1, 0].set_title('碰撞统计')
    axes[1, 0].set_ylabel('回合数')

    # 成功率
    axes[1, 1].pie([results['success_rate'], 1-results['success_rate']],
                   labels=['成功', '失败'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    axes[1, 1].set_title('任务成功率')

    # 性能指标汇总
    axes[1, 2].axis('off')
    summary_text = ".2f"".2f"".2f"".2f"".2f"".2f"f"""
    性能指标汇总 (4奖励系统):
    平均奖励: {results['avg_reward']:.2f}
    领航者奖励: {results['avg_leader_reward']:.2f}
    跟随者奖励: {results['avg_follower_reward']:.2f}
    成功率: {results['success_rate']:.2f}
    平均碰撞: {results['avg_collisions']:.2f}
    奖励组成: 成功/碰撞/进度/避障
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(f"agent/log/CTDE_evaluation_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

def compare_models(env, ppo_agent, model_paths, model_names, num_episodes=10):
    """比较不同模型的性能"""
    all_results = {}

    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n=== 评估模型: {model_name} ===")
        results = evaluate_model(env, ppo_agent, model_path, num_episodes)
        all_results[model_name] = results

        # 保存结果到JSON
        with open(f"agent/log/CTDE_evaluation_{model_name}.json", 'w') as f:
            json.dump(results, f, indent=2)

    # 打印比较结果
    print("\n" + "="*60)
    print("模型性能比较:")
    print("="*60)
    print("<15")
    print("-" * 60)

    for model_name, results in all_results.items():
        print("<15")

    return all_results

def main():
    print("============================================================================================")
    print("CTDE模型评估：领航者-跟随者编队控制系统")
    print("============================================================================================")

    # 创建环境
    env = DroneNavigationMulti(
        num_drones=5,
        use_depth_camera=True,
        depth_camera_range=10.0,
        depth_resolution=16
    )

    # 创建CTDE代理
    ppo_agent = CTDE_PPO(
        leader_state_dim=env.observation_space.shape[0],
        follower_state_dim=env.observation_space.shape[0] - env.depth_feature_dim,
        leader_visual_dim=env.depth_feature_dim,
        action_dim=2,  # 每个无人机的动作维度：2 [thrust, torque] - 平面控制
        num_drones=5,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=100,
        eps_clip=0.2,
        has_continuous_action_space=True,
        action_std_init=0.6
    )

    # 模型路径
    model_paths = [
        "agent/model/CTDE_leader_only.pth",
        "agent/model/CTDE_full_formation.pth",
        "agent/model/CTDE_final_tuned.pth"
    ]

    model_names = [
        "领航者单独训练",
        "编队训练",
        "最终微调"
    ]

    # 检查模型文件是否存在
    available_models = []
    available_names = []
    for path, name in zip(model_paths, model_names):
        if os.path.exists(path):
            available_models.append(path)
            available_names.append(name)
        else:
            print(f"警告: 模型文件不存在 - {path}")

    if not available_models:
        print("错误: 未找到任何模型文件")
        return

    # 比较模型性能
    results = compare_models(env, ppo_agent, available_models, available_names, num_episodes=10)

    # 为每个模型生成图表
    for model_name in available_names:
        if model_name in results:
            plot_evaluation_results(results[model_name], model_name)

    env.close()

    print("============================================================================================")
    print("CTDE模型评估完成！")
    print("============================================================================================")

if __name__ == '__main__':
    main()
