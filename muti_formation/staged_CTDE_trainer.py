
"""
分阶段CTDE PPO训练脚本 - 领航者-跟随者编队控制
第一阶段：训练领航者单独避障和导航
第二阶段：固定领航者，训练跟随者编队跟踪
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from datetime import datetime
import json
import torch
import numpy as np
from agent.CTDE_PPO_agent import CTDE_PPO, device
from drone_envs.envs.drone_env_multi import DroneNavigationMulti
from drone_envs.config import multi_drone_env

# PPO配置 - 🔧 优化超参数，降低探索噪声和更新频率
ppo_config = {
    'lr_actor': 0.0003,      # 降低学习率，提高稳定性
    'lr_critic': 0.0006,     # Critic学习率略高于Actor
    'gamma': 0.99,
    'K_epochs': 10,          # 🔥 从20降到10，标准PPO配置
    'eps_clip': 0.2,
    'has_continuous_action_space': True,
    'action_std_init': 0.3   # 🔥 从0.03降低到0.3，减少探索噪声
}

def save_log_to_json(log, env_name, phase):
    filename = Path(f"agent/log/CTDE_PPO_{env_name}_{phase}_LOG.json")
    filename.parent.mkdir(parents=True, exist_ok=True)
    print(f"saving log to {filename}")
    with open(filename, 'w') as f:
        json.dump(log, f)

def train_leader_only(env, ppo_agent, max_episodes=150, max_steps=500):
    """第一阶段：训练领航者单独避障和导航（增强深度感知版本）- 针对大环境优化"""
    print("=== 第一阶段：训练领航者单独避障和导航 ===")

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        leader_reward_total = 0
        obstacle_detections = 0  # 障碍物检测次数

        for step in range(max_steps):
            # 提取领航者的观测（根据平面模式调整维度）
            leader_obs_dim = 2 + 2 + 4 + 2 + env.depth_feature_dim if env.enforce_planar else 3 + 3 + 4 + 3 + env.depth_feature_dim
            leader_obs = state[:leader_obs_dim]  # 领航者观测是前N维
            # 提取增强深度特征（包含避障决策信息）
            # 在新的观测格式中，深度特征在13-31位置（19维）
            depth_features = leader_obs[13:32] if env.use_leader_camera else None
            
            # 从环境获取额外的避障决策信息用于监控
            if hasattr(env, 'depth_obstacle_processor') and env.use_leader_camera:
                try:
                    # 使用屏蔽后的深度图像进行避障检测，避免无人机自身被误认为障碍物
                    depth_image = env._get_masked_leader_depth()
                    if depth_image is not None and depth_image.size > 0:
                        # 重要：先预处理深度图像再处理
                        raw_depth = depth_image if len(depth_image.shape) == 2 else depth_image[:, :, 0]
                        processed_depth = env.depth_obstacle_processor.preprocess_depth_image(raw_depth)
                        obstacle_detected, min_depth = env.depth_obstacle_processor.detect_obstacles(processed_depth)
                        if obstacle_detected:
                            obstacle_detections += 1
                except:
                    pass

            # 第一阶段：只有领航者动作（3维），跟随者由环境内部处理
            leader_action = ppo_agent.select_action([leader_obs], depth_features)[0]
            combined_actions = leader_action  # 只发送领航者的动作

            next_state, reward, terminated, truncated, info = env.step(combined_actions)

            # 只计算领航者的奖励 - 适应新的4奖励系统
            reward_info = info.get('reward_info', {})
            leader_reward = reward  # 总奖励即领航者奖励
            leader_reward_total += leader_reward

            # 存储领航者的奖励（select_action 已经存储了其他经验）
            ppo_agent.buffers[0].rewards.append(leader_reward)
            ppo_agent.buffers[0].is_terminals.append(terminated or truncated)

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        # 更新领航者策略
        if len(ppo_agent.buffers[0].rewards) > 0:
            ppo_agent.update()

        # 输出增强的训练信息
        print(f"领航者训练回合 {episode + 1}/{max_episodes} | 奖励: {episode_reward:.2f} | 领航者奖励: {leader_reward_total:.2f} | 障碍物检测: {obstacle_detections}")
        
        # 每10个回合输出详细避障统计
        if (episode + 1) % 10 == 0:
            print(f"  - 障碍物检测次数: {obstacle_detections}")

    # 保存领航者模型
    leader_model_path = "agent/model/CTDE_leader_enhanced.pth"
    ppo_agent.save(leader_model_path)
    print(f"增强领航者模型已保存: {leader_model_path}")

    return leader_model_path

def train_followers_only(env, ppo_agent, leader_model_path, max_episodes=150, max_steps=500):
    """第二阶段：固定领航者，训练跟随者编队跟踪 - 针对大环境优化"""
    print("=== 第二阶段：固定领航者，训练跟随者编队跟踪（大环境优化） ===")

    # 切换到第二阶段：启用编队奖励
    env.training_stage = 2
    print("环境已切换到第二阶段：启用跟随者编队奖励")

    # 加载领航者模型
    ppo_agent.load(leader_model_path)
    print(f"加载领航者模型: {leader_model_path}")

    # 冻结领航者网络参数
    for param in ppo_agent.leader_policy.parameters():
        param.requires_grad = False
    for param in ppo_agent.leader_policy_old.parameters():
        param.requires_grad = False

    print("领航者网络已冻结，开始训练跟随者...")

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        follower_reward_total = 0

        for step in range(max_steps):
            # 提取领航者的观测（根据平面模式调整维度）
            leader_obs_dim = 2 + 2 + 4 + 2 + env.depth_feature_dim if env.enforce_planar else 3 + 3 + 4 + 3 + env.depth_feature_dim
            leader_obs = state[:leader_obs_dim]  # 领航者观测是前N维
            # 提取领航者的深度特征（领航者观测中的深度部分，索引13:32）
            depth_features = leader_obs[13:32] if env.use_leader_camera else None

            # 领航者和跟随者都动作
            actions = ppo_agent.select_action([leader_obs] + [state[i*leader_obs_dim:(i+1)*leader_obs_dim] for i in range(1, env.num_drones)], depth_features)
            combined_actions = np.concatenate(actions)

            next_state, reward, terminated, truncated, info = env.step(combined_actions)

            # 分离奖励 - 适应新的4奖励系统
            reward_info = info.get('reward_info', {})
            leader_reward = reward  # 总奖励作为领航者奖励
            follower_reward = 0.0   # 跟随者奖励设为0
            follower_reward_total += follower_reward

            # 存储所有智能体的奖励（select_action 已经存储了其他经验）
            for i in range(env.num_drones):
                if i == 0:  # 领航者
                    ppo_agent.buffers[i].rewards.append(leader_reward)
                else:  # 跟随者
                    ppo_agent.buffers[i].rewards.append(follower_reward)
                ppo_agent.buffers[i].is_terminals.append(terminated or truncated)

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        # 只更新跟随者策略
        if any(len(buffer.rewards) > 0 for buffer in ppo_agent.buffers[1:]):
            ppo_agent.update()

        print(f"跟随者训练回合 {episode + 1}/{max_episodes} | 奖励: {episode_reward:.2f} | 跟随者奖励: {follower_reward_total:.2f}")

    # 保存完整模型
    final_model_path = "agent/model/CTDE_full_formation.pth"
    ppo_agent.save(final_model_path)
    print(f"完整编队模型已保存: {final_model_path}")

    return final_model_path

def joint_fine_tuning(env, ppo_agent, model_path, max_episodes=50, max_steps=300):
    """第三阶段：联合微调所有智能体"""
    print("=== 第三阶段：联合微调所有智能体 ===")

    # 加载模型
    ppo_agent.load(model_path)

    # 解冻领航者网络
    for param in ppo_agent.leader_policy.parameters():
        param.requires_grad = True
    for param in ppo_agent.leader_policy_old.parameters():
        param.requires_grad = True

    print("开始联合微调...")

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 提取领航者的观测（根据平面模式调整维度）
            leader_obs_dim = 2 + 2 + 4 + 2 + env.depth_feature_dim if env.enforce_planar else 3 + 3 + 4 + 3 + env.depth_feature_dim
            leader_obs = state[:leader_obs_dim]  # 领航者观测是前N维
            # 提取领航者的深度特征（领航者观测中的深度部分，索引13:32）
            depth_features = leader_obs[13:32] if env.use_leader_camera else None

            # 所有智能体动作
            actions = ppo_agent.select_action([leader_obs] + [state[i*leader_obs_dim:(i+1)*leader_obs_dim] for i in range(1, env.num_drones)], depth_features)
            combined_actions = np.concatenate(actions)

            next_state, reward, terminated, truncated, info = env.step(combined_actions)

            # 使用完整奖励 - 适应新的4奖励系统
            reward_info = info.get('reward_info', {})
            leader_reward = reward  # 总奖励作为领航者奖励
            follower_reward = 0.0   # 跟随者奖励设为0

            # 存储经验（select_action 已经存储了其他经验）
            for i in range(env.num_drones):
                if i == 0:
                    ppo_agent.buffers[i].rewards.append(leader_reward)
                else:
                    ppo_agent.buffers[i].rewards.append(follower_reward)
                ppo_agent.buffers[i].is_terminals.append(terminated or truncated)

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        # 更新所有策略
        ppo_agent.update()

        print(f"联合微调回合 {episode + 1}/{max_episodes} | 奖励: {episode_reward:.2f}")

    # 保存最终模型
    final_model_path = "agent/model/CTDE_final_tuned.pth"
    ppo_agent.save(final_model_path)
    print(f"最终微调模型已保存: {final_model_path}")

    return final_model_path

def main():
    print("============================================================================================")
    print("开始分阶段CTDE训练：领航者-跟随者架构（集成4奖励系统）")
    print("============================================================================================")

    # 创建环境 - 第一阶段训练：仅训练领航者导航
    env = DroneNavigationMulti(
        num_drones=5,
        use_depth_camera=True,
        depth_camera_range=10.0,
        depth_resolution=16,  # 基础深度特征维度
        enable_formation_force=False,  # 第一阶段禁用编队力，让跟随者悬停
        training_stage=1,  # 第一阶段：取消跟随者编队奖励
        max_steps=5000  # 设置最大步数，避免过早截断
    )
    
    # 打印环境信息
    print(f"环境配置:")
    print(f"  - 无人机数量: {env.num_drones}")
    print(f"  - 观测空间: {env.observation_space.shape}")
    print(f"  - 动作空间: {env.action_space.shape}")
    print(f"  - 深度特征维度: {env.depth_feature_dim}")
    print(f"  - 增强深度特征: 连续深度避障")

    # 创建CTDE代理 - 支持增强深度特征，根据平面模式调整状态维度
    # 从配置文件获取深度特征维度
    from drone_envs.config import multi_drone_env
    leader_visual_dim = multi_drone_env.get("depth_feature_dim", 130)
    
    # 根据平面模式计算状态维度
    # 平面模式: 位置(2) + 速度(2) + 朝向(4) + 目标相对位置(2) + 深度特征
    # 3D模式: 位置(3) + 速度(3) + 朝向(4) + 目标相对位置(3) + 深度特征
    base_state_dim = 2 + 2 + 4 + 2 + env.depth_feature_dim if env.enforce_planar else 3 + 3 + 4 + 3 + env.depth_feature_dim
    
    ppo_agent = CTDE_PPO(
        leader_state_dim=base_state_dim,
        follower_state_dim=base_state_dim,
        leader_visual_dim=leader_visual_dim,  # CNN深度特征维度（从配置文件获取）
        action_dim=2,  # 每个无人机的动作维度：2 [thrust, torque] for body-frame control with camera
        num_drones=5,
        lr_actor=ppo_config['lr_actor'],
        lr_critic=ppo_config['lr_critic'],
        gamma=ppo_config['gamma'],
        K_epochs=ppo_config['K_epochs'],
        eps_clip=ppo_config['eps_clip'],
        has_continuous_action_space=ppo_config['has_continuous_action_space'],
        action_std_init=ppo_config['action_std_init']
    )

    print("CTDE代理已创建，支持增强深度避障特征")
    print("============================================================================================")

    # 第一阶段：训练领航者
    leader_model_path = train_leader_only(env, ppo_agent, max_episodes=500, max_steps=5000)

    # 第二阶段：训练跟随者
    formation_model_path = train_followers_only(env, ppo_agent, leader_model_path, max_episodes=500, max_steps=5000)

    # 第三阶段：联合微调
    final_model_path = joint_fine_tuning(env, ppo_agent, formation_model_path, max_episodes=30, max_steps=300)

    # 保存训练日志
    log = {
        "training_phases": ["leader_enhanced", "followers_only", "joint_fine_tuning"],
        "models": {
            "leader_enhanced": leader_model_path,
            "formation": formation_model_path,
            "final": final_model_path
        },
        "enhancements": {
            "reward_system": "4-reward_minimal",
            "depth_obstacle_avoidance": True,
            "continuous_depth_rewards": True,
            "planar_action_mapping": True,
            "reward_components": ["success", "crash", "progress", "obstacle"]
        },
        "completion_time": str(datetime.now().replace(microsecond=0))
    }
    save_log_to_json(log, "DroneNavigationMultiFormation-v0", "staged_enhanced_training")

    env.close()

    print("============================================================================================")
    print("分阶段CTDE训练完成！（集成4奖励系统）")
    print(f"增强领航者模型: {leader_model_path}")
    print(f"编队模型: {formation_model_path}")
    print(f"最终模型: {final_model_path}")
    print("增强功能: 4奖励系统、连续深度避障、平面动作映射")
    print("============================================================================================")

if __name__ == '__main__':
    main()
