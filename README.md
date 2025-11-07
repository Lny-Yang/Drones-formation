# Pybullet-Gym-Drone - 强化学习无人机导航与编队

基于 [PyBullet](https://pybullet.org/) + [OpenAI Gym](https://github.com/openai/gym) + [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) 的无人机自主导航仿真项目。实现了单机导航、多机编队、分阶段训练等功能。


## 项目特色

- **多种RL算法**: 支持PPO、TRPO (原生实现 + Stable-Baselines3)
- **单机/多机场景**: 单无人机导航 + 多无人机编队飞行
- **分阶段训练**: Leader Phase 1 → Follower Formation → CTDE (Centralized Training Decentralized Execution)
- **室内环境**: 带障碍物的3D室内场景仿真
- **NaN防护**: SafeActorCriticPolicy包装器,防止训练崩溃

## 训练结果

### 单机导航 (Legacy)
| 成功率 | DroneNavigationV0 | DroneNavigationV1 |
|--------|-------------------|-------------------|
| TRPO   | 50.0%             | 39.2%             |
| PPO    | 91.2%             | 11.2%             |

### 多机编队 (最新)
- **Leader Phase 1 (SB3)**: 99,000 episodes, 64% 成功率
- **训练中**: 继续向 200,000 episodes 目标推进

## 快速开始

### 环境配置

```bash
# 创建Python环境
conda create -n drone_sim python=3.8
conda activate drone_sim

# 安装依赖
pip install -r requirements.txt
```

### 单机训练

```bash
# PPO训练
python PPO_trainer.py

# TRPO训练
python TRPO_trainer.py

# 运行已训练模型
python run.py --model PPO --version 0
```

### 多机编队训练

```bash
# Phase 1: Leader单机导航 (SB3)
python muti_formation/train_phase1_SB3.py

# 分阶段CTDE训练
python muti_formation/staged_CTDE_trainer.py

# 评估CTDE模型
python muti_formation/evaluate_CTDE_models.py

# 运行多机编队演示
python muti_formation/run_multi.py
```

### 配置GUI显示

修改 `drone_envs/config.py` 或 `muti_formation/drone_envs/config.py`:

```python
"display": p.GUI  # 打开可视化界面
"display": p.DIRECT  # 关闭可视化 (加速训练)
```

## 项目结构

```
├── agent/                  # 单机RL代理 (PPO/TRPO原生实现)
│   ├── model/             # 训练好的模型
│   ├── log/               # 训练日志
│   └── PPOagent.py
├── muti_formation/        # 多机编队相关
│   ├── agent/             # 多机RL代理
│   │   ├── model_SB3/    # SB3模型检查点
│   │   └── log_SB3/      # 训练数据
│   ├── drone_envs/       # 多机环境定义
│   ├── train_phase1_SB3.py        # Phase 1训练脚本
│   ├── staged_CTDE_trainer.py     # 分阶段CTDE训练
│   └── evaluate_CTDE_models.py    # 模型评估
├── drone_envs/            # 单机环境定义
│   ├── envs/             # 环境版本 (v0, v1, multi)
│   └── resources/        # 3D模型资源 (URDF)
└── README.md
```

## 技术栈

- **仿真引擎**: PyBullet 3.2+
- **强化学习**: Stable-Baselines3 2.0+, PyTorch 1.10+
- **环境框架**: OpenAI Gym / Gymnasium
- **可视化**: Matplotlib, TensorBoard

## 训练技巧

1. **NaN问题**: 使用 `SafeActorCriticPolicy` 包装器 (见 `train_phase1_SB3.py`)
2. **学习率**: 长期训练建议 3e-5 (默认 3e-4 可能导致不稳定)
3. **梯度裁剪**: `max_grad_norm=2.0` 提高稳定性
4. **继续训练**: 脚本自动从检查点恢复训练进度

## 参考资源

- [Medium: Creating OpenAI Gym Environments with PyBullet](https://medium.com/@gerardmaggiolino/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/)

## 许可证

本项目遵循原项目许可证。
