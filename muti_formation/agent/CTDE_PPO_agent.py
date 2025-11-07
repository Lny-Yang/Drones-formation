from datetime import datetime
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque
import random

# 兼容两种导入方式
try:
    from ..drone_envs.config import multi_drone_env  # 相对导入（从根目录运行时）
except ImportError:
    from drone_envs.config import multi_drone_env  # 直接导入（从muti_formation目录运行时）

# 检查是否有可用的 CUDA 设备，如果有则使用 GPU 进行计算，同时清空 CUDA 缓存；否则使用 CPU。
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

################################## CTDE PPO Policy ##################################

# 经验回放缓冲区类，用于存储智能体与环境交互产生的数据
class RolloutBuffer:
    def __init__(self):
        # 初始化一个空的经验回放缓冲区，用于存储智能体在环境中交互产生的数据。
        self.actions = []  # 存储智能体执行的动作
        self.states = []  # 存储智能体所处的状态
        self.logprobs = []  # 存储动作的对数概率
        self.rewards = []  # 存储获得的奖励
        self.state_values = []  # 存储状态值
        self.is_terminals = []  # 存储回合是否结束的标志

    def clear(self):
        # 清空缓冲区中的所有数据
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区 - 专门存储高价值经验"""
    def __init__(self, max_size=1000, success_priority=10.0):
        """
        Args:
            max_size: 缓冲区最大容量
            success_priority: 成功经验的优先级权重
        """
        self.max_size = max_size
        self.success_priority = success_priority
        
        # 使用deque实现固定大小的FIFO缓冲区
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        
        # 统计信息
        self.total_added = 0
        self.success_count = 0
    
    def add_episode(self, states, actions, logprobs, rewards, state_values, is_terminals, episode_return, is_success):
        """
        添加一整个episode的经验
        
        Args:
            states, actions, logprobs, rewards, state_values, is_terminals: episode数据
            episode_return: episode总回报
            is_success: 是否成功到达目标
        """
        # 计算优先级：成功episode获得高优先级，否则基于回报
        if is_success:
            priority = self.success_priority * (1.0 + episode_return / 1000.0)
            self.success_count += 1
        else:
            # 非成功episode，优先级基于归一化回报
            priority = max(0.1, episode_return / 1000.0)  # 最小优先级0.1
        
        # 存储episode数据
        episode_data = {
            'states': states.copy(),
            'actions': actions.copy(),
            'logprobs': logprobs.copy(),
            'rewards': rewards.copy(),
            'state_values': state_values.copy(),
            'is_terminals': is_terminals.copy(),
            'episode_return': episode_return,
            'is_success': is_success,
            'length': len(states)
        }
        
        self.buffer.append(episode_data)
        self.priorities.append(priority)
        self.total_added += 1
    
    def sample(self, num_episodes=5):
        """
        根据优先级采样episodes
        
        Args:
            num_episodes: 采样的episode数量
        
        Returns:
            采样的episodes数据
        """
        if len(self.buffer) == 0:
            return None
        
        # 计算采样数量（不超过缓冲区大小）
        num_episodes = min(num_episodes, len(self.buffer))
        
        # 归一化优先级为概率
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # 根据优先级采样（无放回）
        sampled_indices = np.random.choice(
            len(self.buffer), 
            size=num_episodes, 
            replace=False,
            p=probabilities
        )
        
        # 收集采样的episodes
        sampled_states = []
        sampled_actions = []
        sampled_logprobs = []
        sampled_rewards = []
        sampled_state_values = []
        sampled_is_terminals = []
        
        for idx in sampled_indices:
            episode = self.buffer[idx]
            sampled_states.extend(episode['states'])
            sampled_actions.extend(episode['actions'])
            sampled_logprobs.extend(episode['logprobs'])
            sampled_rewards.extend(episode['rewards'])
            sampled_state_values.extend(episode['state_values'])
            sampled_is_terminals.extend(episode['is_terminals'])
        
        return {
            'states': sampled_states,
            'actions': sampled_actions,
            'logprobs': sampled_logprobs,
            'rewards': sampled_rewards,
            'state_values': sampled_state_values,
            'is_terminals': sampled_is_terminals,
            'num_episodes': num_episodes,
            'total_steps': len(sampled_states)
        }
    
    def __len__(self):
        return len(self.buffer)
    
    def get_stats(self):
        """获取缓冲区统计信息"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'success_count': 0,
                'success_rate': 0.0,
                'avg_priority': 0.0,
                'total_added': self.total_added
            }
        
        success_in_buffer = sum(1 for ep in self.buffer if ep['is_success'])
        
        return {
            'size': len(self.buffer),
            'success_count': success_in_buffer,
            'success_rate': success_in_buffer / len(self.buffer),
            'avg_priority': np.mean(self.priorities),
            'total_added': self.total_added
        }

class VisualFeatureExtractor(nn.Module):
    """改进版视觉特征提取器，用于处理CNN深度特征（128维 + 2额外特征 = 130维）"""
    def __init__(self, input_channels=128, enhanced_channels=2, feature_dim=64):
        super(VisualFeatureExtractor, self).__init__()
        self.input_channels = input_channels  # CNN特征：128维
        self.enhanced_channels = enhanced_channels  # 额外特征：2维
        self.total_channels = input_channels + enhanced_channels  # 总共130维

        # CNN特征处理器（处理128维CNN特征）
        self.cnn_feature_processor = nn.Sequential(
            nn.Linear(input_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

        # 增强特征处理器（处理障碍物检测 + 最小深度）
        self.enhanced_feature_processor = nn.Sequential(
            nn.Linear(enhanced_channels, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim//2),
            nn.ReLU(),
        )

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim + feature_dim//2, feature_dim),
            nn.ReLU(),
        )

        # 避障决策增强器
        self.avoidance_enhancer = nn.Sequential(
            nn.Linear(enhanced_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Sigmoid()  # 输出避障权重
        )

        self.feature_dim = feature_dim

    def forward(self, depth_features):
        """
        处理CNN深度特征输入
        输入: depth_features (batch_size, 130) - 128维CNN特征 + 2维增强特征
        输出: 提取的视觉特征 (batch_size, feature_dim)
        """
        # depth_features: [batch_size, total_channels] 或 [total_channels]
        # total_channels = input_channels(128) + enhanced_channels(2) = 130
        if len(depth_features.shape) == 1:
            depth_features = depth_features.unsqueeze(0)

        batch_size = depth_features.size(0)

        # 分离CNN特征和增强特征
        cnn_features = depth_features[:, :self.input_channels]  # 前128维：CNN特征
        enhanced_features = depth_features[:, self.input_channels:self.total_channels]  # 后2维：增强特征

        # 处理CNN特征
        cnn_processed = self.cnn_feature_processor(cnn_features)

        # 处理增强特征
        enhanced_processed = self.enhanced_feature_processor(enhanced_features)

        # 生成避障权重
        avoidance_weights = self.avoidance_enhancer(enhanced_features)

        # 特征融合
        combined_features = torch.cat([cnn_processed, enhanced_processed], dim=-1)
        fused_features = self.fusion_layer(combined_features)

        # 应用避障权重增强
        final_features = fused_features * avoidance_weights.mean(dim=1, keepdim=True)

        return final_features

class LeaderActorCritic(nn.Module):
    """领航者专用网络，集成增强视觉输入（包含避障决策信息）"""
    def __init__(self, state_dim, visual_dim, action_dim, has_continuous_action_space, action_std_init):
        super(LeaderActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        self.visual_dim = visual_dim  # 保存视觉维度

        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # 增强视觉特征提取器（支持130维输入：128 CNN特征 + 2增强特征）
        self.visual_extractor = VisualFeatureExtractor(
            input_channels=128,  # CNN深度特征：128维
            enhanced_channels=2,  # 增强特征：障碍检测 + 最小深度
            feature_dim=64
        )

        # 状态编码器（处理非视觉状态）
        # 动态计算非视觉状态维度：总状态维度减去深度特征维度
        # 深度特征固定为130维（128 CNN + 2增强）
        self.depth_feature_dim = 130
        self.non_visual_dim = max(1, state_dim - self.depth_feature_dim)
        
        self.state_encoder = nn.Sequential(
            nn.Linear(self.non_visual_dim, 64),  # 动态非视觉状态维度
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # 融合网络（视觉 + 状态）
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 64, 128),  # 状态特征64 + 视觉特征64
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # 演员网络
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
                # 移除Tanh激活函数，让网络直接输出动作值
                # 动作会在环境中被clip到config定义的范围
            )
            # 🔥 初始化输出层的bias为0，让初始动作接近0
            nn.init.zeros_(self.actor[-1].bias)
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)  # 小的初始权重
        else:
            self.actor = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # 评论家网络（全局状态评估）
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self, state, depth_features):
        # 确保输入是批次格式
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # 添加批次维度
        if depth_features is None or depth_features.numel() == 0:
            # 如果深度特征为空，创建默认特征
            batch_size = state.shape[0]
            depth_features = torch.ones(batch_size, 130, device=state.device)  # 128 CNN + 2增强
        elif len(depth_features.shape) == 1:
            depth_features = depth_features.unsqueeze(0)  # 添加批次维度
        
        # 确保深度特征维度正确
        if depth_features.shape[-1] < 130:
            # 如果特征不足130维，用1.0填充到130维
            batch_size = depth_features.shape[0]
            current_dim = depth_features.shape[-1]
            padding = torch.ones(batch_size, 130 - current_dim, device=depth_features.device)
            depth_features = torch.cat([depth_features, padding], dim=-1)
        elif depth_features.shape[-1] > 130:
            # 如果特征超过130维，截取前130维
            depth_features = depth_features[:, :130]

        # 提取视觉特征
        visual_features = self.visual_extractor(depth_features)

        # 编码状态特征（排除深度特征）
        # 非视觉状态是前non_visual_dim维
        non_visual_features = state[:, :self.non_visual_dim]  # pos + vel + orientation + target
        state_features = self.state_encoder(non_visual_features)

        # 融合视觉和状态特征
        fused_features = torch.cat([state_features, visual_features], dim=-1)
        fused_output = self.fusion_net(fused_features)

        return fused_output

    def act(self, state, depth_features):
        fused_output = self.forward(state, depth_features)

        if self.has_continuous_action_space:
            action_mean = self.actor(fused_output)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(fused_output)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(fused_output)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, depth_features, action):
        fused_output = self.forward(state, depth_features)

        if self.has_continuous_action_space:
            action_mean = self.actor(fused_output)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(fused_output)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(fused_output)

        return action_logprobs, state_values, dist_entropy

class FollowerActorCritic(nn.Module):
    """跟随者专用网络，简化版，支持动态状态维度"""
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(FollowerActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        self.state_dim = state_dim  # 保存状态维度

        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # 跟随者网络（简化，无视觉输入）
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),  # 使用动态状态维度
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # 演员网络
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim)
                # 移除Tanh激活函数
            )
            # 🔥 初始化输出层的bias为0，让初始动作接近0
            nn.init.zeros_(self.actor[-1].bias)
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)  # 小的初始权重
        else:
            self.actor = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.Softmax(dim=-1)
            )

        # 评论家网络
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def act(self, state):
        features = self.network(state)

        if self.has_continuous_action_space:
            action_mean = self.actor(features)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(features)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        features = self.network(state)

        if self.has_continuous_action_space:
            action_mean = self.actor(features)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)

        return action_logprobs, state_values, dist_entropy

class CTDE_PPO:
    """CTDE架构的PPO算法"""
    def __init__(self,
                 leader_state_dim,
                 follower_state_dim,
                 leader_visual_dim,
                 action_dim,
                 num_drones=5,
                 lr_actor=0.0003,
                 lr_critic=0.001,
                 gamma=0.99,
                 K_epochs=40,
                 eps_clip=0.2,
                 has_continuous_action_space=True,
                 action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space
        self.num_drones = num_drones
        self.leader_idx = 0
        self.follower_indices = list(range(1, num_drones))
        self.action_std_init = action_std_init  # 添加这个属性

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # 创建经验回放缓冲区（为每个智能体）
        self.buffers = [RolloutBuffer() for _ in range(num_drones)]
        
        # 🔥 新增：优先经验回放缓冲区 - 防止灾难性遗忘
        # 🎯 修复奖励欺骗后的优化: 进一步扩容以应对20000回合长期训练
        # 容量计算: 20000回合 × 90%成功率 × 2%保留 = 360个成功经验
        # 实际设置: 1000容量可容纳更多历史，更强防遗忘能力
        self.replay_buffer = PrioritizedReplayBuffer(
            max_size=1000,  # � 从500扩大到1000，强化历史经验保留
            success_priority=80.0  # � 从50.0提升到80.0，最大化成功经验权重
        )
        self.use_replay = True  # 是否使用经验回放
        self.replay_ratio = 0.5  # 保持0.5，平衡历史和新经验（避免过拟合历史）

        # 创建领航者和跟随者网络
        self.leader_policy = LeaderActorCritic(
            leader_state_dim, leader_visual_dim, action_dim,
            has_continuous_action_space, action_std_init
        ).to(device)

        self.follower_policies = [
            FollowerActorCritic(follower_state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
            for _ in range(num_drones - 1)
        ]

        # 全局评论家（CTDE的核心）
        # 第一阶段只有一个领航者，使用领航者自己的评论家
        # 第二阶段有多个智能体，使用全局评论家
        if self.num_drones == 1:
            # 第一阶段：领航者评论家已在LeaderActorCritic中定义
            self.global_critic = None
        else:
            # 第二阶段：全局评论家
            self.global_critic = nn.Sequential(
                nn.Linear(leader_state_dim + follower_state_dim * (num_drones - 1), 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ).to(device)

        # 优化器
        self.leader_optimizer = torch.optim.Adam([
            {'params': self.leader_policy.parameters(), 'lr': lr_actor}
        ])

        self.follower_optimizers = [
            torch.optim.Adam([
                {'params': policy.parameters(), 'lr': lr_actor}
            ]) for policy in self.follower_policies
        ]

        # 只有在全局评论家存在时才创建其优化器
        if self.global_critic is not None:
            self.critic_optimizer = torch.optim.Adam([
                {'params': self.global_critic.parameters(), 'lr': lr_critic}
            ])
        else:
            self.critic_optimizer = None

        # 旧网络
        self.leader_policy_old = LeaderActorCritic(
            leader_state_dim, leader_visual_dim, action_dim,
            has_continuous_action_space, action_std_init
        ).to(device)
        self.leader_policy_old.load_state_dict(self.leader_policy.state_dict())

        self.follower_policies_old = []
        for i in range(num_drones - 1):
            old_policy = FollowerActorCritic(
                follower_state_dim, action_dim, has_continuous_action_space, action_std_init
            ).to(device)
            old_policy.load_state_dict(self.follower_policies[i].state_dict())
            self.follower_policies_old.append(old_policy)

        self.MseLoss = nn.MSELoss()

    def select_action(self, states, depth_features=None, leader_only=False):
        """为所有智能体选择动作"""
        actions = []

        # 处理单个状态输入的情况（用于测试）
        if not isinstance(states, list):
            states = [states]

        # 确保有足够的智能体状态
        if len(states) < self.num_drones and not leader_only:
            # 如果状态不足，复制最后一个状态
            while len(states) < self.num_drones:
                states.append(states[-1])

        # 领航者动作选择
        leader_state = states[0]  # 领航者总是第一个

        # 从状态中提取深度特征（如果没有提供depth_features）
        # 深度特征是状态的最后130维
        if depth_features is None and len(leader_state) >= self.leader_policy.depth_feature_dim:
            depth_start_idx = len(leader_state) - self.leader_policy.depth_feature_dim
            depth_features = leader_state[depth_start_idx:]  # 提取最后130维作为深度特征

        if depth_features is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(leader_state).to(device)
                depth_tensor = torch.FloatTensor(depth_features).to(device)
                action, action_logprob, state_val = self.leader_policy_old.act(state_tensor, depth_tensor)

            self.buffers[self.leader_idx].states.append(state_tensor)
            self.buffers[self.leader_idx].actions.append(action)
            self.buffers[self.leader_idx].logprobs.append(action_logprob)
            self.buffers[self.leader_idx].state_values.append(state_val)

            # Clip action to config range
            action_np = action.detach().cpu().numpy().flatten()
            action_np[0] = np.clip(action_np[0], multi_drone_env['thrust_lower_bound'], multi_drone_env['thrust_upper_bound'])
            action_np[1] = np.clip(action_np[1], multi_drone_env['torque_lower_bound'], multi_drone_env['torque_upper_bound'])
            actions.append(action_np)
        else:
            # 如果没有深度特征,使用简化版（将领航者当作跟随者处理）
            with torch.no_grad():
                state_tensor = torch.FloatTensor(leader_state).to(device)
                action, action_logprob, state_val = self.follower_policies_old[0].act(state_tensor)

            self.buffers[self.leader_idx].states.append(state_tensor)
            self.buffers[self.leader_idx].actions.append(action)
            self.buffers[self.leader_idx].logprobs.append(action_logprob)
            self.buffers[self.leader_idx].state_values.append(state_val)

            # Clip action to config range
            action_np = action.detach().cpu().numpy().flatten()
            action_np[0] = np.clip(action_np[0], multi_drone_env['thrust_lower_bound'], multi_drone_env['thrust_upper_bound'])
            action_np[1] = np.clip(action_np[1], multi_drone_env['torque_lower_bound'], multi_drone_env['torque_upper_bound'])
            actions.append(action_np)

        # 如果只需要领航者动作，返回
        if leader_only:
            return actions

        # 跟随者动作选择
        for i, follower_idx in enumerate(self.follower_indices):
            if follower_idx < len(states):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(states[follower_idx]).to(device)
                    # 跟随者观测已经是完整的26维（深度特征用0填充）
                    action, action_logprob, state_val = self.follower_policies_old[i].act(state_tensor)

                self.buffers[follower_idx].states.append(state_tensor)
                self.buffers[follower_idx].actions.append(action)
                self.buffers[follower_idx].logprobs.append(action_logprob)
                self.buffers[follower_idx].state_values.append(state_val)

                # Clip action to config range
                action_np = action.detach().cpu().numpy().flatten()
                action_np[0] = np.clip(action_np[0], multi_drone_env['thrust_lower_bound'], multi_drone_env['thrust_upper_bound'])
                action_np[1] = np.clip(action_np[1], multi_drone_env['torque_lower_bound'], multi_drone_env['torque_upper_bound'])
                actions.append(action_np)
            else:
                # 如果没有足够的状态，使用零动作
                actions.append(np.zeros(self.action_dim))

        return actions

    def update(self):
        """CTDE更新策略 - 集成优先经验回放"""
        
        # 🔥 步骤1: 保存当前episode到经验回放缓冲区
        if self.use_replay and len(self.buffers[0].rewards) > 0:
            # 计算episode总回报
            episode_return = sum(self.buffers[0].rewards)
            
            # 判断是否成功（从最后一个奖励判断，成功奖励通常很大）
            is_success = any(r > 1500 for r in self.buffers[0].rewards)  # 成功奖励一般>2000
            
            # 添加到回放缓冲区
            self.replay_buffer.add_episode(
                states=self.buffers[0].states.copy(),
                actions=self.buffers[0].actions.copy(),
                logprobs=self.buffers[0].logprobs.copy(),
                rewards=self.buffers[0].rewards.copy(),
                state_values=self.buffers[0].state_values.copy(),
                is_terminals=self.buffers[0].is_terminals.copy(),
                episode_return=episode_return,
                is_success=is_success
            )
        
        # 🔥 步骤2: 从经验回放缓冲区采样并混合到当前buffer
        if self.use_replay and len(self.replay_buffer) > 10:  # 至少有10个episodes才开始回放
            # 采样历史经验
            replay_data = self.replay_buffer.sample(num_episodes=min(5, len(self.replay_buffer) // 10))
            
            if replay_data is not None:
                # 将回放数据混合到当前buffer（只混合领航者数据）
                # 计算混合比例
                current_size = len(self.buffers[0].rewards)
                replay_size = len(replay_data['rewards'])
                
                # 按replay_ratio比例混合
                target_replay_size = int(current_size * self.replay_ratio / (1 - self.replay_ratio))
                if replay_size > target_replay_size:
                    # 随机采样一部分回放数据
                    indices = random.sample(range(replay_size), target_replay_size)
                    replay_data = {
                        'states': [replay_data['states'][i] for i in indices],
                        'actions': [replay_data['actions'][i] for i in indices],
                        'logprobs': [replay_data['logprobs'][i] for i in indices],
                        'rewards': [replay_data['rewards'][i] for i in indices],
                        'state_values': [replay_data['state_values'][i] for i in indices],
                        'is_terminals': [replay_data['is_terminals'][i] for i in indices],
                    }
                
                # 混合数据到buffer
                self.buffers[0].states.extend(replay_data['states'])
                self.buffers[0].actions.extend(replay_data['actions'])
                self.buffers[0].logprobs.extend(replay_data['logprobs'])
                self.buffers[0].rewards.extend(replay_data['rewards'])
                self.buffers[0].state_values.extend(replay_data['state_values'])
                self.buffers[0].is_terminals.extend(replay_data['is_terminals'])
        
        # 收集所有智能体的奖励
        all_rewards = []
        all_original_rewards = []  # 保存原始回报用于优势函数计算
        all_states = []
        all_actions = []
        all_logprobs = []
        all_state_values = []
        all_is_terminals = []

        for i in range(self.num_drones):
            # 跳过没有奖励数据的智能体
            if len(self.buffers[i].rewards) == 0:
                continue
                
            # Monte Carlo 估计回报
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.buffers[i].rewards),
                                         reversed(self.buffers[i].is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            # 保存原始回报用于优势函数计算
            original_rewards = rewards.clone()
            
            # 奖励标准化：使用更稳定的方法，避免数值问题
            if len(rewards) > 1:  # 至少需要2个样本才能计算方差
                rewards_mean = rewards.mean()
                rewards_std = rewards.std()
                # 只在奖励变化显著时进行标准化（避免过度标准化）
                if rewards_std > max(0.1, abs(rewards_mean) * 0.05):  # 标准差大于平均值的5%或0.1
                    # 使用温和的标准化，保留奖励的相对强度
                    rewards = (rewards - rewards_mean) / (rewards_std + 1e-8)
                    # 限制标准化后的奖励范围，避免极端值
                    rewards = torch.clamp(rewards, -5.0, 5.0)

            all_rewards.append(rewards)
            all_original_rewards.append(original_rewards)  # 保存原始回报
            all_states.append(torch.squeeze(torch.stack(self.buffers[i].states, dim=0)).detach().to(device))
            all_actions.append(torch.squeeze(torch.stack(self.buffers[i].actions, dim=0)).detach().to(device))
            all_logprobs.append(torch.squeeze(torch.stack(self.buffers[i].logprobs, dim=0)).detach().to(device))
            all_state_values.append(torch.squeeze(torch.stack(self.buffers[i].state_values, dim=0)).detach().to(device))
            all_is_terminals.append(self.buffers[i].is_terminals)

        # 如果没有活跃的智能体，返回
        if not all_rewards:
            return

        # 全局状态用于评论家
        if len(all_states) == 1:
            # 第一阶段：只有领航者，使用领航者自己的评论家
            global_input = all_states[0]
            use_global_critic = False
            # 从状态中提取深度特征（最后130维）
            if all_states[0].dim() > 1:  # 批次数据
                depth_start_idx = all_states[0].shape[-1] - self.leader_policy.depth_feature_dim
                depth_features_batch = all_states[0][:, depth_start_idx:].detach().to(device)
            else:  # 单个数据
                depth_start_idx = len(all_states[0]) - self.leader_policy.depth_feature_dim
                depth_features_batch = all_states[0][depth_start_idx:].detach().to(device).unsqueeze(0)
        else:
            # 其他阶段：所有智能体，使用全局评论家
            # 将所有智能体的状态沿着特征维度连接
            global_input = torch.cat(all_states, dim=-1)  # [batch_size, total_state_dim]
            use_global_critic = True
            depth_features_batch = None

        # 更新评论家
        for _ in range(self.K_epochs):
            if use_global_critic:
                # 全局评论家使用所有智能体的联合状态，输出全局价值
                global_values = self.global_critic(global_input)  # [batch_size, 1]
                # 为每个智能体分配相同的全局价值（CTDE的核心思想）
                critic_loss = sum(self.MseLoss(global_values.squeeze(), r) for r in all_rewards)
            else:
                # 第一阶段使用领航者评论家
                fused_features = self.leader_policy(all_states[0], depth_features_batch)
                critic_values = self.leader_policy.critic(fused_features)
                critic_loss = self.MseLoss(critic_values.squeeze(), all_rewards[0])

            if use_global_critic:
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.global_critic.parameters(), max_norm=0.5)  # 梯度裁剪
                self.critic_optimizer.step()
            else:
                # 第一阶段更新领航者评论家（通过leader_optimizer，因为评论家是leader_policy的一部分）
                pass  # 评论家已经通过leader_optimizer更新了

        # 计算优势函数（使用原始回报，不使用标准化后的）
        if use_global_critic:
            # 全局评论家为所有智能体提供相同的全局价值估计
            global_values = self.global_critic(global_input).detach()  # [batch_size, 1]
            # 每个智能体使用相同的全局价值计算优势（CTDE的核心）
            advantages = [original_r - global_values.squeeze() for original_r in all_original_rewards]
        else:
            fused_features = self.leader_policy(all_states[0], depth_features_batch)
            critic_values = self.leader_policy.critic(fused_features).detach()
            advantages = [all_original_rewards[0] - critic_values.squeeze()]

        # 更新领航者策略（如果有领航者数据）
        if len(self.buffers[self.leader_idx].rewards) > 0:
            leader_state_idx = 0  # 领航者是第一个
            for _ in range(self.K_epochs):
                # 从状态中提取深度特征（最后130维）
                if all_states[leader_state_idx].dim() > 1:  # 批次数据
                    depth_start_idx = all_states[leader_state_idx].shape[-1] - self.leader_policy.depth_feature_dim
                    depth_features = all_states[leader_state_idx][:, depth_start_idx:].detach().to(device)
                else:  # 单个数据
                    depth_start_idx = len(all_states[leader_state_idx]) - self.leader_policy.depth_feature_dim
                    depth_features = all_states[leader_state_idx][depth_start_idx:].detach().to(device).unsqueeze(0)

                logprobs, state_values, dist_entropy = self.leader_policy.evaluate(
                    all_states[leader_state_idx], depth_features, all_actions[leader_state_idx])

                ratios = torch.exp(logprobs - all_logprobs[leader_state_idx].detach())
                surr1 = ratios * advantages[leader_state_idx]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[leader_state_idx]

                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), all_rewards[leader_state_idx]) - 0.05 * dist_entropy

                self.leader_optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.leader_policy.parameters(), max_norm=0.5)  # 梯度裁剪
                self.leader_optimizer.step()

        # 更新跟随者策略（如果有跟随者数据）
        for follower_idx in range(len(self.follower_indices)):
            actual_idx = self.follower_indices[follower_idx]
            if len(self.buffers[actual_idx].rewards) > 0:
                state_idx = [i for i, idx in enumerate(range(self.num_drones)) if len(self.buffers[idx].rewards) > 0].index(actual_idx)
                for _ in range(self.K_epochs):
                    logprobs, state_values, dist_entropy = self.follower_policies[follower_idx].evaluate(
                        all_states[state_idx], all_actions[state_idx])

                    ratios = torch.exp(logprobs - all_logprobs[state_idx].detach())
                    surr1 = ratios * advantages[state_idx]
                    surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[state_idx]

                    loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), all_rewards[state_idx]) - 0.05 * dist_entropy

                    self.follower_optimizers[follower_idx].zero_grad()
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(self.follower_policies[follower_idx].parameters(), max_norm=0.5)  # 梯度裁剪
                    self.follower_optimizers[follower_idx].step()

        # 更新旧网络
        if len(self.buffers[self.leader_idx].rewards) > 0:
            self.leader_policy_old.load_state_dict(self.leader_policy.state_dict())
        for i in range(len(self.follower_policies)):
            actual_idx = self.follower_indices[i]
            if len(self.buffers[actual_idx].rewards) > 0:
                self.follower_policies_old[i].load_state_dict(self.follower_policies[i].state_dict())

        # 清空缓冲区
        for buffer in self.buffers:
            buffer.clear()

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """动态衰减动作标准差以提高采样效率"""
        if self.has_continuous_action_space:
            self.action_std = max(min_action_std, self.action_std * action_std_decay_rate)
            self.leader_policy.set_action_std(self.action_std)
            for policy in self.follower_policies:
                policy.set_action_std(self.action_std)
            print(f"动作标准差更新为: {self.action_std}")
    
    def set_action_std(self, new_std):
        """设置新的动作标准差"""
        if self.has_continuous_action_space:
            self.action_std = new_std
            self.leader_policy.set_action_std(self.action_std)
            for policy in self.follower_policies:
                policy.set_action_std(self.action_std)
    
    def get_replay_buffer_stats(self):
        """获取经验回放缓冲区统计信息"""
        return self.replay_buffer.get_stats()

    def save(self, checkpoint_path):
        """保存模型"""
        checkpoint = {
            'leader_policy': self.leader_policy_old.state_dict(),
            'follower_policies': [policy.state_dict() for policy in self.follower_policies_old],
        }
        # 只有在全局评论家存在时才保存
        if self.global_critic is not None:
            checkpoint['global_critic'] = self.global_critic.state_dict()
        
        torch.save(checkpoint, checkpoint_path)

    def validate_algorithm(self):
        """验证PPO算法实现的正确性"""
        print("=== PPO算法验证 ===")

        # 验证网络结构
        print("✓ 领航者网络参数:", sum(p.numel() for p in self.leader_policy.parameters()))
        print("✓ 跟随者网络数量:", len(self.follower_policies))
        if self.global_critic is not None:
            print("✓ 全局评论家参数:", sum(p.numel() for p in self.global_critic.parameters()))

        # 验证动作选择
        test_state = np.random.randn(156)  # 假设状态维度为156（26 + 130）
        test_depth = np.random.randn(130)  # 深度特征

        try:
            actions = self.select_action([test_state], test_depth, leader_only=True)
            print(f"✓ 动作选择成功: {len(actions)} 个动作")
        except Exception as e:
            print(f"✗ 动作选择失败: {e}")
            return False

        # 验证缓冲区
        for i, buffer in enumerate(self.buffers):
            if len(buffer.states) > 0:
                print(f"✓ 智能体{i}缓冲区有数据: {len(buffer.states)} 条")

        # 验证优化器
        print("✓ 领航者优化器:", type(self.leader_optimizer).__name__)
        print("✓ 跟随者优化器数量:", len(self.follower_optimizers))

        print("✓ PPO算法验证完成")
        return True
