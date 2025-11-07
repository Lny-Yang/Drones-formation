"""
从TensorBoard日志中提取训练指标

用于提取之前训练的详细指标数据
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("❌ 需要安装tensorboard: pip install tensorboard")
    exit(1)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def extract_tensorboard_data(log_dir):
    """从TensorBoard日志中提取数据"""
    
    # 查找所有事件文件
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    
    if not event_files:
        print(f"❌ 未找到TensorBoard日志文件: {log_dir}")
        return None
    
    print(f"找到 {len(event_files)} 个TensorBoard日志文件")
    
    # 选择最新的日志文件
    latest_event_file = max(event_files, key=lambda x: x.stat().st_mtime)
    print(f"使用最新日志: {latest_event_file.parent.name}")
    
    # 加载事件数据
    ea = event_accumulator.EventAccumulator(str(latest_event_file.parent))
    ea.Reload()
    
    # 提取所有可用的标量
    tags = ea.Tags()['scalars']
    print(f"\n可用的指标标签: {len(tags)}个")
    
    metrics_data = {}
    for tag in tags:
        try:
            events = ea.Scalars(tag)
            metrics_data[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events],
                'wall_times': [e.wall_time for e in events]
            }
        except Exception as e:
            print(f"⚠️  无法提取 {tag}: {e}")
    
    return metrics_data

def convert_to_training_metrics(tensorboard_data):
    """将TensorBoard数据转换为训练指标格式"""
    
    if not tensorboard_data:
        return []
    
    # 获取步数列表（使用最长的序列）
    all_steps = []
    for tag, data in tensorboard_data.items():
        all_steps.extend(data['steps'])
    unique_steps = sorted(set(all_steps))
    
    print(f"\n总共 {len(unique_steps)} 个训练步数记录")
    
    # 为每个步数构建指标字典
    training_metrics = []
    
    for step in unique_steps:
        metrics = {
            'step': int(step),
            'train': {},
            'time': {},
            'rollout': {}
        }
        
        # 提取该步数的所有指标
        for tag, data in tensorboard_data.items():
            if step in data['steps']:
                idx = data['steps'].index(step)
                value = data['values'][idx]
                
                # 分类存储
                if tag.startswith('train/'):
                    key = tag.replace('train/', '')
                    metrics['train'][key] = float(value)
                elif tag.startswith('time/'):
                    key = tag.replace('time/', '')
                    metrics['time'][key] = float(value)
                elif tag.startswith('rollout/'):
                    key = tag.replace('rollout/', '')
                    metrics['rollout'][key] = float(value)
                else:
                    metrics[tag] = float(value)
        
        if metrics['train']:  # 只添加有训练数据的记录
            training_metrics.append(metrics)
    
    return training_metrics

def plot_tensorboard_metrics(metrics_data, save_path):
    """绘制TensorBoard指标"""
    
    # 提取关键指标
    train_tags = [tag for tag in metrics_data.keys() if tag.startswith('train/')]
    
    if not train_tags:
        print("⚠️  没有训练指标可绘制")
        return
    
    # 创建图表
    n_plots = min(len(train_tags), 9)
    n_rows = (n_plots + 2) // 3
    n_cols = min(3, n_plots)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, tag in enumerate(train_tags[:9]):
        ax = axes[i]
        data = metrics_data[tag]
        
        ax.plot(data['steps'], data['values'], alpha=0.7)
        ax.set_xlabel('步数')
        ax.set_ylabel('值')
        ax.set_title(tag.replace('train/', ''))
        ax.grid(True, alpha=0.3)
        
        # 对某些指标使用对数尺度
        if 'loss' in tag.lower():
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {save_path}")

def main():
    """主函数"""
    log_dir = Path(__file__).parent / "agent" / "log_SB3" / "tensorboard"
    
    if not log_dir.exists():
        print(f"❌ TensorBoard日志目录不存在: {log_dir}")
        return
    
    print("="*80)
    print("从TensorBoard日志中提取训练指标")
    print("="*80)
    
    # 提取数据
    tensorboard_data = extract_tensorboard_data(log_dir)
    
    if not tensorboard_data:
        return
    
    # 转换格式
    training_metrics = convert_to_training_metrics(tensorboard_data)
    
    # 保存为JSON
    output_dir = Path(__file__).parent / "agent" / "log_SB3"
    json_path = output_dir / "training_metrics_from_tensorboard.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(training_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 训练指标已保存: {json_path}")
    print(f"   共 {len(training_metrics)} 条记录")
    
    # 绘制图表
    plot_path = output_dir / "training_metrics_from_tensorboard.png"
    plot_tensorboard_metrics(tensorboard_data, plot_path)
    
    # 打印统计信息
    print("\n可用的指标标签:")
    for tag in sorted(tensorboard_data.keys()):
        n_points = len(tensorboard_data[tag]['steps'])
        print(f"  - {tag}: {n_points} 个数据点")
    
    print("="*80)
    print("提取完成！")
    print("="*80)

if __name__ == '__main__':
    main()
