"""
批量测试多个checkpoint模型
用于找出训练过程中的最佳模型
"""
import subprocess
import sys
from pathlib import Path
import json
import pandas as pd

def test_checkpoint(model_path, episodes=50):
    """测试单个checkpoint"""
    print(f"\n{'='*80}")
    print(f"测试模型: {model_path}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        "muti_formation/test_phase1_model.py",  
        "--model", str(model_path),
        "--episodes", str(episodes),
        "--no_render"  # 批量测试时不渲染
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败: {e}")
        return False

def collect_results():
    """收集所有测试结果"""
    # 🔧 从专门的测试结果文件夹读取
    log_dir = Path("muti_formation/agent/log/test_results")
    results = []
    
    if not log_dir.exists():
        print(f"⚠️ 测试结果文件夹不存在: {log_dir}")
        return results
    
    for result_file in log_dir.glob("phase1_test_results_*.json"):
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        model_name = result_file.stem.replace("phase1_test_results_", "")
        
        # 提取关键指标
        num_episodes = len(data['episode_rewards'])
        success_rate = data['success_count'] / num_episodes if num_episodes > 0 else 0
        collision_rate = data['collision_count'] / num_episodes if num_episodes > 0 else 0
        avg_reward = sum(data['episode_rewards']) / num_episodes if num_episodes > 0 else 0
        avg_length = sum(data['episode_lengths']) / num_episodes if num_episodes > 0 else 0
        
        # 计算成功奖励占比
        success_ratio = 0
        if data.get('success_episode_rewards'):
            avg_success_reward = sum(data['success_episode_rewards']) / len(data['success_episode_rewards'])
            success_component = sum([r for r in data['reward_components']['success'] if r > 0])
            if success_component > 0:
                success_ratio = (success_component / len([r for r in data['reward_components']['success'] if r > 0])) / avg_success_reward
        
        # 计算成功/碰撞比
        success_fail_ratio = 0
        if data.get('success_episode_rewards') and data.get('collision_episode_rewards'):
            avg_success = sum(data['success_episode_rewards']) / len(data['success_episode_rewards'])
            avg_collision = sum(data['collision_episode_rewards']) / len(data['collision_episode_rewards'])
            if avg_collision > 0:
                success_fail_ratio = avg_success / avg_collision
        
        results.append({
            'model': model_name,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'success_ratio': success_ratio,
            'success_fail_ratio': success_fail_ratio
        })
    
    return results

def generate_comparison_report(results):
    """生成对比报告"""
    df = pd.DataFrame(results)
    df = df.sort_values('success_rate', ascending=False)
    
    print("\n" + "="*120)
    print("📊 模型对比分析报告")
    print("="*120)
    
    print("\n按成功率排序:")
    print("-"*120)
    print(f"{'模型名称':<40} | {'成功率':>8} | {'碰撞率':>8} | {'平均奖励':>10} | {'平均步数':>8} | {'成功占比':>8} | {'成功/碰撞比':>10}")
    print("-"*120)
    
    for _, row in df.iterrows():
        print(f"{row['model']:<40} | "
              f"{row['success_rate']:>7.1%} | "
              f"{row['collision_rate']:>7.1%} | "
              f"{row['avg_reward']:>10.2f} | "
              f"{row['avg_length']:>8.1f} | "
              f"{row['success_ratio']:>7.1%} | "
              f"{row['success_fail_ratio']:>10.2f}")
    
    print("-"*120)
    
    # 找出最佳模型
    best_model = df.iloc[0]
    print(f"\n🏆 最佳模型: {best_model['model']}")
    print(f"   成功率: {best_model['success_rate']:.1%}")
    print(f"   平均奖励: {best_model['avg_reward']:.2f}")
    print(f"   平均步数: {best_model['avg_length']:.1f}")
    print(f"   成功奖励占比: {best_model['success_ratio']:.1%}")
    print(f"   成功/碰撞比: {best_model['success_fail_ratio']:.2f}:1")
    
    # 🔧 保存报告到测试结果文件夹
    test_results_dir = Path("muti_formation/agent/log/test_results")
    test_results_dir.mkdir(parents=True, exist_ok=True)
    report_path = test_results_dir / "model_comparison_report.csv"
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n📄 对比报告已保存: {report_path}")
    print("="*120)

def main():
    """主函数"""
    print("🚀 开始批量测试模型")
    
    # 定义要测试的checkpoint
    model_dir = Path("muti_formation/agent/model")
    
    # 方式1: 测试所有checkpoint
    # checkpoints = sorted(model_dir.glob("leader_phase1_episode_*.pth"))
    
    # 方式2: 测试特定checkpoint（推荐）
    episodes_to_test = [9000, 11000]  # 🔧 新增：测试14000回合模型
    checkpoints = []
    for ep in episodes_to_test:
        if ep == 'final':
            checkpoint_path = model_dir / "leader_phase1_final.pth"
        else:
            checkpoint_path = model_dir / f"leader_phase1_episode_{ep}.pth"
        
        if checkpoint_path.exists():
            checkpoints.append(checkpoint_path)
        else:
            print(f"⚠️ 模型不存在: {checkpoint_path}")
    
    print(f"找到 {len(checkpoints)} 个模型待测试")
    
    # 测试每个checkpoint
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"\n进度: [{i}/{len(checkpoints)}]")
        test_checkpoint(checkpoint, episodes=50)
    
    # 收集并分析结果
    print("\n" + "="*80)
    print("📊 收集测试结果...")
    print("="*80)
    results = collect_results()
    
    if results:
        generate_comparison_report(results)
    else:
        print("❌ 未找到测试结果")

if __name__ == '__main__':
    main()
