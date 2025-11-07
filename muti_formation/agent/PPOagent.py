from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np
import gym

# 检查是否有可用的 CUDA 设备，如果有则使用 GPU 进行计算，同时清空 CUDA 缓存；否则使用 CPU。
# 设置设备为 CPU 或 CUDA
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")

################################## PPO Policy ##################################

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


# 演员 - 评论家网络类，包含演员网络和评论家网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        # 标记是否为连续动作空间
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            # 连续动作空间的动作维度
            self.action_dim = action_dim
            # 初始化动作方差，将其移动到指定设备上
            #调用 PyTorch 的 full 函数创建一个长度为 action_dim 的一维张量，张量中每个元素的值都为 action_std_init * action_std_init。action_std_init 是初始动作标准差，这里通过平方得到动作方差。
            #将创建好的张量移动到之前设置的计算设备（CPU 或 GPU）上，确保后续计算能在该设备上进行。最终将这个张量赋值给类的实例属性 self.action_var。
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # 演员网络
        if has_continuous_action_space :
            # 连续动作空间的演员网络结构
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),  # 全连接层，输入维度为状态维度，输出维度为 64
                            nn.Tanh(),  # 双曲正切激活函数
                            nn.Linear(64, 64),  # 全连接层，输入输出维度均为 64
                            nn.Tanh(),  # 双曲正切激活函数
                            nn.Linear(64, action_dim),  # 全连接层，输入维度为 64，输出维度为动作维度
                            nn.Tanh()  # 双曲正切激活函数
                            #连续动作空间里，动作可以在一个连续的范围内取值，例如机器人关节的角度、无人机的飞行速度等。网络需要输出具体的动作值，因此最后使用 nn.Tanh() 激活函数。
                            #nn.Tanh() 函数会把输出值映射到 [-1, 1] 区间，这能有效限制动作值的范围，防止输出值过大或过小，符合连续动作空间中动作值通常有一定范围限制的需求。
                        )
        else:
            # 离散动作空间的演员网络结构
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),  # 全连接层，输入维度为状态维度，输出维度为 64
                            nn.Tanh(),  # 双曲正切激活函数
                            nn.Linear(64, 64),  # 全连接层，输入输出维度均为 64
                            nn.Tanh(),  # 双曲正切激活函数
                            nn.Linear(64, action_dim),  # 全连接层，输入维度为 64，输出维度为动作维度
                            nn.Softmax(dim=-1)  # 对最后一个维度进行 Softmax 操作，输出动作概率分布
                            #nn.Softmax(dim=-1) 函数会对最后一个维度的输入进行归一化处理，将输出转换为概率分布，每个元素代表对应动作的选择概率，且所有元素之和为 1，符合离散动作空间中选择某个动作的概率特性。
                            #通过输出概率分布，智能体可以依据这些概率进行采样，从而选择合适的动作。
                        )

        
        # 评论家网络
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),  # 全连接层，输入维度为状态维度，输出维度为 64
                        nn.Tanh(),  # 双曲正切激活函数
                        nn.Linear(64, 64),  # 全连接层，输入输出维度均为 64
                        nn.Tanh(),  # 双曲正切激活函数
                        nn.Linear(64, 1)  # 全连接层，输入维度为 64，输出维度为 1，用于估计状态值
                        #这段代码构建了一个包含两个隐藏层的多层感知机作为评论家网络。
                        #网络接收状态向量作为输入，经过一系列的线性变换和非线性激活，最终输出一个标量值，表示该状态的价值。
                        # 在 PPO 算法中，这个价值估计会用于计算优势函数，帮助演员网络更新策略。
                    )
        
    def set_action_std(self, new_action_std):
        # 设置连续动作空间的动作标准差
        #此方法的作用是在连续动作空间下更新动作方差，而在离散动作空间下给出警告信息，避免用户错误调用。更新动作方差能调整动作分布的离散程度，有助于智能体在训练过程中探索不同的动作。
        if self.has_continuous_action_space:
            # 计算新的动作方差并移动到指定设备上
            #这里将新的动作标准差平方，得到新的动作方差。
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        #在 PyTorch 中，forward 方法用于实现神经网络的前向传播计算。
        # 禁止直接调用 forward 方法
        #在这里，意味着 ActorCritic 类不希望用户直接调用 forward 方法，因为该类的具体前向传播逻辑通过 act 和 evaluate 方法实现。
        raise NotImplementedError
    

    def act(self, state):
        #act 方法根据动作空间类型创建不同的概率分布，从分布中采样得到动作，并计算动作的对数概率和状态值。
        # 这些信息会被用于智能体与环境的交互以及后续的策略更新

        # 根据当前状态选择动作
        if self.has_continuous_action_space:
            # 连续动作空间：通过演员网络得到动作均值
            # 将当前状态 state 输入到演员网络 self.actor 中，得到动作的均值。在连续动作空间里，动作均值是一个连续的值。
            action_mean = self.actor(state)
            # 计算协方差矩阵
            #torch.diag(self.action_var)：将之前初始化的动作方差 self.action_var 转换为对角矩阵，对角线上的元素是各个动作维度的方差。
            #.unsqueeze(dim=0)：在第 0 维上增加一个维度，使协方差矩阵的维度与动作均值匹配。
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # 创建多元正态分布
            #使用 MultivariateNormal 类创建一个多元正态分布，其均值为 action_mean，协方差矩阵为 cov_mat。
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            # 离散动作空间：通过演员网络得到动作概率分布
            action_probs = self.actor(state)
            # 创建分类分布
            dist = Categorical(action_probs)

        # 从分布中采样得到动作
        action = dist.sample()
        # 计算动作的对数概率
        action_logprob = dist.log_prob(action)
        # 通过评论家网络得到状态值
        #将当前状态 state 输入到评论家网络 self.critic 中，得到该状态的价值估计。
        state_val = self.critic(state)

        #detach() 方法用于将张量从计算图中分离出来，避免在后续计算中对这些张量进行梯度计算。最后返回采样动作、动作的对数概率以及状态值。
        return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action):
        # 评估给定状态和动作
        if self.has_continuous_action_space:
            # 连续动作空间：通过演员网络得到动作均值
            action_mean = self.actor(state)
            # 扩展动作方差以匹配动作均值的维度
            action_var = self.action_var.expand_as(action_mean)
            # 计算协方差矩阵并移动到指定设备上
            cov_mat = torch.diag_embed(action_var).to(device)
            # 创建多元正态分布
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # 对于单动作连续环境，调整动作维度
            #如果动作维度为 1，将 action 重新调整形状为 (-1, 1)，以确保维度符合后续计算要求。
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            # 离散动作空间：通过演员网络得到动作概率分布
            action_probs = self.actor(state)
            # 创建分类分布
            dist = Categorical(action_probs)

        # 计算动作的对数概率
        action_logprobs = dist.log_prob(action)
        # 计算策略熵
        dist_entropy = dist.entropy()
        # 通过评论家网络得到状态值
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


# PPO 算法主类
class PPO:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 lr_actor = 0.0003, 
                 lr_critic = 0.001, 
                 gamma = 0.99, 
                 K_epochs = 100, 
                 eps_clip = 0.2, 
                 has_continuous_action_space = True, 
                 action_std_init=0.6):
        # 标记是否为连续动作空间
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            # 连续动作空间的初始动作标准差
            self.action_std = action_std_init

        # 折扣因子
        self.gamma = gamma
        # PPO 算法的裁剪系数
        self.eps_clip = eps_clip
        # 策略更新的迭代次数
        self.K_epochs = K_epochs
        
        # 创建经验回放缓冲区
        self.buffer = RolloutBuffer()

        # 创建演员 - 评论家网络并移动到指定设备上
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # 创建优化器，分别设置演员网络和评论家网络的学习率
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        # 创建旧的演员 - 评论家网络并移动到指定设备上
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # 将新网络的参数复制到旧网络
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 均方误差损失函数
        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):
        # 设置连续动作空间的动作标准差
        if self.has_continuous_action_space:
            # 更新动作标准差
            self.action_std = new_action_std
            # 更新新网络的动作标准差
            self.policy.set_action_std(new_action_std)
            # 更新旧网络的动作标准差
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            # 衰减动作标准差
            self.action_std = self.action_std - action_std_decay_rate
            # 保留四位小数
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                # 如果动作标准差小于等于最小值，设置为最小值
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            # 设置新的动作标准差
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):
        # 根据当前状态选择动作
        if self.has_continuous_action_space:
            #上下文管理器，用于临时禁用梯度计算，减少计算开销，因为选择动作时不需要进行反向传播。
            with torch.no_grad():
                # 将状态转换为张量并移动到指定设备上
                state = torch.FloatTensor(state).to(device)
                # 通过旧网络选择动作、计算动作对数概率和状态值
                action, action_logprob, state_val = self.policy_old.act(state)

            # 将数据存入经验回放缓冲区
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            #将动作张量从计算图中分离出来，移动到 CPU 上，再转换为 NumPy 数组。 将 NumPy 数组展平为一维数组后返回。
            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                # 将状态转换为张量并移动到指定设备上
                state = torch.FloatTensor(state).to(device)
                # 通过旧网络选择动作、计算动作对数概率和状态值
                action, action_logprob, state_val = self.policy_old.act(state)
            
            # 将数据存入经验回放缓冲区
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            # 返回动作的标量值
            return action.item()


    def update(self):
        # 更新策略网络

        # Monte Carlo 估计回报
        rewards = []
        discounted_reward = 0
        # 逆序遍历经验回放缓冲区中的奖励和回合结束标志。
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                # 如果回合结束，重置折扣回报
                discounted_reward = 0
            # 计算折扣回报
            discounted_reward = reward + (self.gamma * discounted_reward)
            # 将折扣回报插入到列表开头
            rewards.insert(0, discounted_reward)
            
        # 归一化奖励
        #将 rewards 列表转换为 PyTorch 张量，并移动到指定设备（CPU 或 GPU）
        #对奖励张量进行归一化处理，减去均值并除以标准差（加 1e-7 防止除零错误）。
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 将列表转换为张量
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # 计算优势函数,优势函数衡量了某个动作相较于平均动作的优劣，通过折扣回报减去旧的状态值得到。
        advantages = rewards.detach() - old_state_values.detach()
        

        # 优化策略 K 个回合
        #循环 self.K_epochs 次更新策略网络。
        for _ in range(self.K_epochs):

            # 评估旧动作和状态值
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 匹配状态值张量的维度与奖励张量的维度, 压缩 state_values 张量维度。
            state_values = torch.squeeze(state_values)
            
            # 计算概率比 (pi_theta / pi_theta__old) 计算新策略和旧策略的概率比。
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算两个替代损失 surr1 和 surr2，surr2 对概率比进行裁剪。
            surr1 = ratios * advantages  #其中 ratios 是新策略和旧策略的概率比，advantages 是优势函数，代表某个动作相较于平均动作的优劣。surr1 是直接用概率比乘以优势函数得到的损失。
            #torch.clamp 函数将 ratios 限制在 [1 - eps_clip, 1 + eps_clip] 范围内，避免概率比过大或过小，eps_clip 是 PPO 算法的裁剪系数。
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # 计算 PPO 裁剪目标的最终损失,包含替代损失、状态值估计误差和策略熵。
            #取 surr1 和 surr2 中的较小值，PPO 算法通过这种方式限制策略更新的步长，防止策略更新幅度过大。
            #-torch.min: 因为在优化过程中，我们要最大化策略的收益，而 PyTorch 的优化器默认是最小化损失，所以加上负号将最大化问题转换为最小化问题。
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # 梯度清零
            self.optimizer.zero_grad()
            # 反向传播计算梯度
            loss.mean().backward()
            # 更新参数
            self.optimizer.step()
            
        # 将新网络的参数复制到旧网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空经验回放缓冲区
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        # 保存旧策略网络的参数到指定路径
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load_model(self, checkpoint_path):
        # 从指定路径加载模型参数到新旧策略网络
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    
    # 兼容别名：与脚本中的 ppo_agent.load(path) 保持一致
    def load(self, checkpoint_path):
        self.load_model(checkpoint_path)
        
        
       

