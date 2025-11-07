# 测试结果文件夹

本文件夹专门存放模型测试结果，包括单模型测试和批量对比测试的所有输出文件。

## 📂 文件结构

```
test_results/
├── README.md                                    # 本说明文件
├── phase1_test_results_<model_name>.json        # 单模型测试的JSON数据
├── phase1_test_results_<model_name>.png         # 单模型测试的可视化图表（6个子图）
└── model_comparison_report.csv                  # 批量测试的对比报告
```

## 📊 文件说明

### 1. JSON数据文件 (`phase1_test_results_*.json`)
包含详细的测试数据：
- `episode_rewards`: 每轮的总奖励
- `episode_lengths`: 每轮的步数
- `success_count`: 成功次数
- `collision_count`: 碰撞次数
- `reward_components`: 8个奖励分量的详细数据
  - `success`: 成功奖励（3000分）
  - `crash`: 碰撞惩罚（-100分）
  - `navigation`: 导航奖励（距离接近奖励）
  - `forward_movement`: 前进运动奖励
  - `obstacle_avoidance`: 避障奖励
  - `rotation_guidance`: 旋转引导奖励
  - `adaptive_speed`: 自适应速度奖励
  - `step_penalty`: 步数惩罚（-0.5×step/3000）
- `success_episode_rewards`: 成功轮次的奖励列表
- `collision_episode_rewards`: 碰撞轮次的奖励列表

**用途**: 用于深度分析模型性能，可编程解析进行自定义统计分析。

### 2. PNG图表文件 (`phase1_test_results_*.png`)
6个子图的可视化分析：
1. **Episode Rewards**: 每轮奖励趋势（蓝色）+ 移动平均（橙色）
2. **Episode Lengths**: 每轮步数趋势
3. **Episode Results**: 成功（绿色）vs 碰撞（红色）分布
4. **Reward Distribution**: 奖励值分布直方图
5. **Reward Components**: 8个奖励分量的平均值柱状图
6. **Success vs Collision**: 成功轮次和碰撞轮次的奖励对比箱线图

**用途**: 快速可视化评估模型性能，直观展示各奖励分量贡献。

### 3. CSV对比报告 (`model_comparison_report.csv`)
批量测试多个checkpoint的对比数据：
- `model`: 模型名称
- `success_rate`: 成功率
- `collision_rate`: 碰撞率
- `avg_reward`: 平均奖励
- `avg_length`: 平均步数
- `success_ratio`: 成功奖励占比（验证奖励系统健康度）
- `success_fail_ratio`: 成功/碰撞比（验证无奖励欺骗）

**用途**: 对比不同训练阶段的checkpoint，找出最佳模型。

## 🎯 关键验证指标

### ✅ 健康模型的标准
| 指标 | 目标值 | 含义 |
|------|--------|------|
| **成功率** | ≥92% | 模型完成任务能力强 |
| **成功奖励占比** | 77-87% | success_bonus（3000分）主导奖励系统 |
| **成功/碰撞比** | ≥5:1 | 成功奖励远大于失败，无奖励欺骗风险 |
| **平均步数** | 65-75 | 高效导航，不拖延 |
| **平均奖励** | 3400-3450 | 综合性能指标 |

### ⚠️ 需要关注的情况
- 成功率 < 90%: 可能需要更多训练或参数调整
- 成功奖励占比 < 70%: 奖励系统可能存在问题
- 成功/碰撞比 < 3:1: 可能存在奖励欺骗风险
- 平均步数 > 100: 导航效率低，可能在拖延
- 平均步数 < 50: 可能过于激进，导致碰撞

## 📖 使用示例

### 单模型测试
```bash
# 测试最终模型
python muti_formation/test_phase1_model.py

# 测试特定checkpoint
python muti_formation/test_phase1_model.py --model leader_phase1_episode_15000.pth

# 快速测试（20轮，无渲染）
python muti_formation/test_phase1_model.py --episodes 20 --no_render
```

生成文件：
- `test_results/phase1_test_results_leader_phase1_final.json`
- `test_results/phase1_test_results_leader_phase1_final.png`

### 批量测试
```bash
python muti_formation/batch_test_models.py
```

生成文件：
- `test_results/phase1_test_results_leader_phase1_episode_*.json` (多个)
- `test_results/phase1_test_results_leader_phase1_episode_*.png` (多个)
- `test_results/model_comparison_report.csv` (汇总对比)

## 🔍 数据分析建议

1. **查看最终模型表现**:
   - 打开 `phase1_test_results_leader_phase1_final.png` 查看6图分析
   - 重点关注成功率和成功奖励占比

2. **对比不同训练阶段**:
   - 打开 `model_comparison_report.csv`
   - 按成功率排序，找出最佳checkpoint

3. **验证奖励系统健康**:
   - 检查 JSON 中的 `reward_components`
   - 计算 `success_ratio = sum(success成功轮) / sum(total成功轮)`
   - 应在 77-87% 范围内

4. **排查性能问题**:
   - 如果成功率低：查看碰撞类型分布
   - 如果奖励异常：检查各奖励分量贡献
   - 如果步数过长：可能存在拖延行为

## 🗂️ 文件清理

测试结果文件可能随时间积累较多，建议定期清理：

```bash
# 删除所有测试结果（慎用！）
rm -rf muti_formation/agent/log/test_results/*

# 只保留最近的测试结果
# 手动删除旧的 phase1_test_results_* 文件
```

## 📌 注意事项

1. **文件命名规则**: 所有文件名中的 `<model_name>` 来自模型文件的文件名（不含 `.pth` 扩展名）
2. **编码格式**: JSON文件使用UTF-8编码，CSV文件使用UTF-8-BOM编码（兼容Excel）
3. **数据完整性**: 每次测试会覆盖同名文件，如需保留历史记录请手动重命名
4. **大文件处理**: PNG图表以300 DPI保存，文件较大但画质清晰，适合论文使用

---

📅 最后更新: 2025-10-18
