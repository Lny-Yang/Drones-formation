"""
批量测试多个SB3 PPO checkpoint模型
用于找出训练过程中的最佳模型
"""
import subprocess
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

def test_checkpoint(model_path, episodes=50):
    """测试单个SB3 checkpoint"""
    print(f"\n{'='*80}")
    print(f"测试模型: {model_path}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        "muti_formation/test_phase1_SB3.py",
        "--model", str(model_path),
        "--episodes", str(episodes),
        "--no_render",  # 批量测试时不渲染
        "--save_dir", "muti_formation/agent/log_SB3/test_results"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败: {e}")
        return False

def collect_results():
    """收集所有SB3测试结果"""
    log_dir = Path("muti_formation/agent/log_SB3")
    results = []
    
    if not log_dir.exists():
        print(f"⚠️ 测试结果文件夹不存在: {log_dir}")
        return results
    
    for result_file in log_dir.glob("phase1_test_results_*.json"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            model_name = result_file.stem.replace("phase1_test_results_", "")
            
            # 提取关键指标
            num_episodes = len(data['episode_rewards'])
            if num_episodes == 0:
                continue
            
            success_rate = data['success_count'] / num_episodes
            collision_rate = data['collision_count'] / num_episodes
            timeout_rate = data['timeout_count'] / num_episodes
            boundary_collision_rate = data['boundary_collision_count'] / num_episodes
            physical_collision_rate = data['physical_collision_count'] / num_episodes
            
            avg_reward = np.mean(data['episode_rewards'])
            std_reward = np.std(data['episode_rewards'])
            avg_length = np.mean(data['episode_lengths'])
            
            # 平均最小深度和目标距离
            avg_min_depth = np.mean(data['min_depths']) if data.get('min_depths') else 0
            avg_goal_distance = np.mean(data['goal_distances']) if data.get('goal_distances') else 0
            
            # 奖励分量分析
            reward_components = data.get('reward_components', {})
            avg_success_reward = np.mean(reward_components.get('success', [0]))
            avg_crash_reward = np.mean(reward_components.get('crash', [0]))
            avg_dense_reward = np.mean(reward_components.get('dense', [0]))
            
            # 计算成功时的平均奖励和碰撞时的平均奖励
            success_episode_rewards = [r for i, r in enumerate(data['episode_rewards']) 
                                      if i < len(data['episode_rewards']) and data['success_count'] > 0]
            collision_episode_rewards = [r for i, r in enumerate(data['episode_rewards']) 
                                        if i < len(data['episode_rewards']) and data['collision_count'] > 0]
            
            avg_success_episode_reward = np.mean(success_episode_rewards) if success_episode_rewards else 0
            avg_collision_episode_reward = np.mean(collision_episode_rewards) if collision_episode_rewards else 0
            
            # 成功/碰撞比
            success_fail_ratio = avg_success_episode_reward / avg_collision_episode_reward if avg_collision_episode_reward != 0 else 0
            
            results.append({
                'model': model_name,
                'episodes': num_episodes,
                'success_rate': success_rate,
                'collision_rate': collision_rate,
                'timeout_rate': timeout_rate,
                'boundary_collision_rate': boundary_collision_rate,
                'physical_collision_rate': physical_collision_rate,
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'avg_length': avg_length,
                'avg_min_depth': avg_min_depth,
                'avg_goal_distance': avg_goal_distance,
                'avg_success_reward': avg_success_reward,
                'avg_crash_reward': avg_crash_reward,
                'avg_dense_reward': avg_dense_reward,
                'avg_success_episode_reward': avg_success_episode_reward,
                'avg_collision_episode_reward': avg_collision_episode_reward,
                'success_fail_ratio': success_fail_ratio
            })
        except Exception as e:
            print(f"⚠️ 处理文件 {result_file} 时出错: {e}")
            continue
    
    return results

def generate_comparison_report(results):
    """生成对比报告"""
    if not results:
        print("❌ 没有可用的测试结果")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values('success_rate', ascending=False)
    
    print("\n" + "="*140)
    print("📊 SB3 PPO模型对比分析报告")
    print("="*140)
    
    print("\n🏆 按成功率排序:")
    print("-"*140)
    print(f"{'模型名称':<45} | {'回合':>5} | {'成功率':>8} | {'碰撞率':>8} | {'超时率':>8} | {'平均奖励':>10} | {'奖励标准差':>10} | {'平均步数':>8}")
    print("-"*140)
    
    for _, row in df.iterrows():
        print(f"{row['model']:<45} | "
              f"{row['episodes']:>5d} | "
              f"{row['success_rate']:>7.1%} | "
              f"{row['collision_rate']:>7.1%} | "
              f"{row['timeout_rate']:>7.1%} | "
              f"{row['avg_reward']:>10.2f} | "
              f"{row['std_reward']:>10.2f} | "
              f"{row['avg_length']:>8.1f}")
    
    print("-"*140)
    
    # 详细分析表
    print("\n📈 详细分析:")
    print("-"*140)
    print(f"{'模型名称':<45} | {'边界碰撞':>9} | {'物理碰撞':>9} | {'最小深度':>9} | {'目标距离':>9} | {'成功/碰撞比':>12}")
    print("-"*140)
    
    for _, row in df.iterrows():
        print(f"{row['model']:<45} | "
              f"{row['boundary_collision_rate']:>8.1%} | "
              f"{row['physical_collision_rate']:>8.1%} | "
              f"{row['avg_min_depth']:>8.2f}m | "
              f"{row['avg_goal_distance']:>8.2f}m | "
              f"{row['success_fail_ratio']:>11.2f}:1")
    
    print("-"*140)
    
    # 奖励分量分析
    print("\n💰 奖励分量分析:")
    print("-"*140)
    print(f"{'模型名称':<45} | {'成功奖励':>10} | {'碰撞惩罚':>10} | {'密集奖励':>10} | {'成功回合奖励':>12} | {'碰撞回合奖励':>12}")
    print("-"*140)
    
    for _, row in df.iterrows():
        print(f"{row['model']:<45} | "
              f"{row['avg_success_reward']:>10.2f} | "
              f"{row['avg_crash_reward']:>10.2f} | "
              f"{row['avg_dense_reward']:>10.2f} | "
              f"{row['avg_success_episode_reward']:>12.2f} | "
              f"{row['avg_collision_episode_reward']:>12.2f}")
    
    print("-"*140)
    
    # 找出最佳模型
    best_model = df.iloc[0]
    print(f"\n🏆 最佳模型（按成功率）: {best_model['model']}")
    print(f"   测试回合: {best_model['episodes']}")
    print(f"   成功率: {best_model['success_rate']:.1%}")
    print(f"   碰撞率: {best_model['collision_rate']:.1%}")
    print(f"   超时率: {best_model['timeout_rate']:.1%}")
    print(f"   平均奖励: {best_model['avg_reward']:.2f} ± {best_model['std_reward']:.2f}")
    print(f"   平均步数: {best_model['avg_length']:.1f}")
    print(f"   平均最小深度: {best_model['avg_min_depth']:.2f}m")
    print(f"   平均目标距离: {best_model['avg_goal_distance']:.2f}m")
    print(f"   成功/碰撞比: {best_model['success_fail_ratio']:.2f}:1")
    
    # 找出平均奖励最高的模型
    best_reward_model = df.loc[df['avg_reward'].idxmax()]
    if best_reward_model['model'] != best_model['model']:
        print(f"\n💎 最高平均奖励模型: {best_reward_model['model']}")
        print(f"   平均奖励: {best_reward_model['avg_reward']:.2f}")
        print(f"   成功率: {best_reward_model['success_rate']:.1%}")
    
    # 找出最稳定的模型（标准差最小）
    best_stable_model = df.loc[df['std_reward'].idxmin()]
    if best_stable_model['model'] not in [best_model['model'], best_reward_model['model']]:
        print(f"\n⚖️  最稳定模型（标准差最小）: {best_stable_model['model']}")
        print(f"   奖励标准差: {best_stable_model['std_reward']:.2f}")
        print(f"   成功率: {best_stable_model['success_rate']:.1%}")
    
    # 保存报告
    log_dir = Path("muti_formation/agent/log_SB3")
    report_path = log_dir / "model_comparison_report.csv"
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n📄 详细对比报告已保存: {report_path}")
    
    # 保存简化版报告（只包含关键指标）
    simplified_df = df[['model', 'episodes', 'success_rate', 'collision_rate', 'avg_reward', 
                        'avg_length', 'avg_min_depth', 'success_fail_ratio']]
    simplified_report_path = log_dir / "model_comparison_summary.csv"
    simplified_df.to_csv(simplified_report_path, index=False, encoding='utf-8-sig')
    print(f"📄 简化报告已保存: {simplified_report_path}")
    
    print("="*140)

def main():
    """主函数"""
    print("🚀 开始批量测试SB3 PPO模型")
    
    # 定义要测试的checkpoint
    model_dir = Path("muti_formation/agent/model_SB3")
    
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return
    
    # 方式1: 测试所有checkpoint
    # checkpoints = sorted(model_dir.glob("leader_phase1_episode_*.zip"))
    
    # 方式2: 测试特定checkpoint（推荐）
    episodes_to_test = [30000, 35000, 40000, 45000, 50000, 55000, 'final']  # 🔧 根据需要调整
    checkpoints = []
    for ep in episodes_to_test:
        if ep == 'final':
            checkpoint_path = model_dir / "leader_phase1_final"
        else:
            checkpoint_path = model_dir / f"leader_phase1_episode_{ep}"
        
        # 检查是否存在（不含.zip后缀，SB3会自动添加）
        if (checkpoint_path.parent / (checkpoint_path.name + '.zip')).exists():
            checkpoints.append(checkpoint_path)
        else:
            print(f"⚠️ 模型不存在: {checkpoint_path}.zip")
    
    if not checkpoints:
        print("❌ 未找到任何模型文件")
        print(f"请确保模型文件位于: {model_dir}")
        print("模型文件命名格式: leader_phase1_episode_*.zip 或 leader_phase1_final.zip")
        return
    
    print(f"找到 {len(checkpoints)} 个模型待测试")
    print(f"模型列表: {[c.name for c in checkpoints]}")
    
    # 测试每个checkpoint
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"\n进度: [{i}/{len(checkpoints)}]")
        test_checkpoint(checkpoint, episodes=200)  # 🔧 可调整测试回合数
    
    # 收集并分析结果
    print("\n" + "="*80)
    print("📊 收集测试结果...")
    print("="*80)
    results = collect_results()
    
    if results:
        generate_comparison_report(results)
    else:
        print("❌ 未找到测试结果")
        print("请检查测试是否成功完成，以及结果文件是否保存在 log_SB3/ 目录")

if __name__ == '__main__':
    main()
