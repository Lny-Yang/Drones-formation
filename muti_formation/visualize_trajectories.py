"""
无人机轨迹可视化脚本
支持多种轨迹可视化方式，专门为大量训练数据优化
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

class TrajectoryVisualizer:
    """轨迹可视化器 - 支持多种可视化模式"""

    def __init__(self, trajectory_files: List[str]):
        """初始化可视化器

        Args:
            trajectory_files: 轨迹数据文件路径列表
        """
        self.trajectory_files = trajectory_files if isinstance(trajectory_files, list) else [trajectory_files]
        self.data = None
        self.trajectories = []

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 室内环境边界（根据环境配置）
        self.env_bounds = {
            'x_min': -14.5, 'x_max': 14.5,
            'y_min': -14.5, 'y_max': 14.5
        }

        self.load_data()

    def load_data(self):
        """加载轨迹数据"""
        self.trajectories = []
        total_episodes = 0
        
        for trajectory_file in self.trajectory_files:
            try:
                with open(trajectory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.trajectories.extend(data['trajectories'])
                    total_episodes += len(data['trajectories'])
                    print(f"成功加载轨迹数据: {trajectory_file} - {len(data['trajectories'])} 个回合")
            except Exception as e:
                print(f"加载轨迹数据失败: {trajectory_file} - {e}")
                continue
        
        # 如果有数据，设置 summary 为最后一个文件的 summary
        if self.trajectories:
            self.data = {'trajectories': self.trajectories, 'summary': data.get('summary', {})}
            print(f"总共加载轨迹数据: {total_episodes} 个回合")
        else:
            self.data = None
        return self.data is not None

    def plot_single_trajectory(self, episode_id: int, save_path: Optional[str] = None):
        """绘制单个回合的轨迹

        Args:
            episode_id: 回合ID
            save_path: 保存路径，如果为None则显示图像
        """
        traj_data = None
        for traj in self.trajectories:
            if traj['episode_id'] == episode_id:
                traj_data = traj
                break

        if traj_data is None:
            print(f"未找到回合 {episode_id} 的轨迹数据")
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # 绘制环境边界
        self._draw_environment(ax)

        # 绘制轨迹
        trajectory = np.array(traj_data['drone_trajectories'][0])  # 领航者轨迹
        rewards = np.array(traj_data['rewards'])

        # 根据奖励值着色
        norm = plt.Normalize(vmin=np.min(rewards), vmax=np.max(rewards))
        colors = cm.RdYlGn(norm(rewards))  # 红黄绿 colormap

        # 绘制轨迹线
        for i in range(len(trajectory) - 1):
            ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1],
                   color=colors[i], linewidth=2, alpha=0.8)

        # 绘制起点和终点
        start_pos = trajectory[0]
        end_pos = trajectory[-1]
        goal_pos = np.array(traj_data['goal_position'])

        ax.scatter(*start_pos, c='blue', s=100, marker='o', label='起点', zorder=5)
        ax.scatter(*end_pos, c='red', s=100, marker='X', label='终点', zorder=5)
        ax.scatter(goal_pos[0], goal_pos[1], c='green', s=100, marker='*', label='目标', zorder=5)

        # 设置标题和标签
        success_status = "成功" if traj_data['success'] else "失败"
        collision_status = "碰撞" if traj_data['collision'] else "无碰撞"

        ax.set_title(f'回合 {episode_id} 轨迹 - {success_status}({collision_status})\n'
                    f'总奖励: {traj_data["total_reward"]:.2f}, 步数: {traj_data["total_steps"]}')
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cm.RdYlGn, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('奖励值')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"轨迹图已保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_multiple_trajectories(self, episode_ids: List[int], max_trajectories: int = 20,
                                 save_path: Optional[str] = None):
        """绘制多个回合的轨迹叠加

        Args:
            episode_ids: 要绘制的回合ID列表
            max_trajectories: 最大轨迹数量
            save_path: 保存路径
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 绘制环境边界
        self._draw_environment(ax)

        # 限制轨迹数量
        if len(episode_ids) > max_trajectories:
            episode_ids = np.random.choice(episode_ids, max_trajectories, replace=False)

        colors = cm.rainbow(np.linspace(0, 1, len(episode_ids)))

        for i, episode_id in enumerate(episode_ids):
            traj_data = None
            for traj in self.trajectories:
                if traj['episode_id'] == episode_id:
                    traj_data = traj
                    break

            if traj_data is None:
                continue

            trajectory = np.array(traj_data['drone_trajectories'][0])
            success = traj_data['success']

            # 成功轨迹用实线，失败用虚线
            linestyle = '-' if success else '--'
            alpha = 0.7 if success else 0.4
            success_status = "成功" if success else "失败"

            ax.plot(trajectory[:, 0], trajectory[:, 1],
                   color=colors[i], linewidth=1.5, linestyle=linestyle,
                   alpha=alpha)

            # 标记起点和终点
            ax.scatter(*trajectory[0], color=colors[i], s=30, marker='o', alpha=alpha)
            ax.scatter(*trajectory[-1], color=colors[i], s=50, marker='X' if success else 'x', alpha=alpha)

        # 绘制所有目标位置（淡化显示）
        for traj in self.trajectories[:50]:  # 只显示前50个避免过于密集
            goal_pos = np.array(traj['goal_position'])
            ax.scatter(goal_pos[0], goal_pos[1], c='gray', s=20, marker='*', alpha=0.1)

        ax.set_title(f'多回合轨迹叠加 (显示最后 {len(episode_ids)} 个回合)')
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"多轨迹图已保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_trajectory_heatmap(self, episode_range: Optional[tuple] = None,
                              save_path: Optional[str] = None):
        """绘制轨迹热力图 - 显示轨迹密度

        Args:
            episode_range: 回合范围 (start, end)，如果为None则使用全部
            save_path: 保存路径
        """
        # 收集所有轨迹点
        all_points = []

        start_idx, end_idx = episode_range if episode_range else (0, len(self.trajectories))

        for traj in self.trajectories[start_idx:end_idx]:
            trajectory = np.array(traj['drone_trajectories'][0])
            all_points.extend(trajectory.tolist())

        if not all_points:
            print("没有轨迹数据可用于热力图")
            return

        all_points = np.array(all_points)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # 绘制环境边界
        self._draw_environment(ax)

        # 创建热力图
        hist, xedges, yedges = np.histogram2d(
            all_points[:, 0], all_points[:, 1],
            bins=50, range=[[self.env_bounds['x_min'], self.env_bounds['x_max']],
                           [self.env_bounds['y_min'], self.env_bounds['y_max']]]
        )

        # 对数变换避免极端值
        hist_log = np.log(hist + 1)

        im = ax.imshow(hist_log.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      cmap='hot', alpha=0.7)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('轨迹密度 (对数)')

        ax.set_title(f'轨迹密度热力图 (回合 {start_idx+1} - {end_idx})')
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热力图已保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_success_vs_failure_analysis(self, save_path: Optional[str] = None):
        """成功 vs 失败轨迹分析

        Args:
            save_path: 保存路径
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 分离成功和失败轨迹
        success_trajectories = []
        failure_trajectories = []

        for traj in self.trajectories:
            trajectory = np.array(traj['drone_trajectories'][0])
            if traj['success']:
                success_trajectories.append(trajectory)
            else:
                failure_trajectories.append(trajectory)

        # 1. 成功率统计
        total_episodes = len(self.trajectories)
        success_rate = len(success_trajectories) / total_episodes * 100

        ax1.pie([success_rate, 100-success_rate], labels=['成功', '失败'],
               autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        ax1.set_title('任务成功率')

        # 2. 轨迹长度分布
        success_lengths = [len(traj) for traj in success_trajectories]
        failure_lengths = [len(traj) for traj in failure_trajectories]

        ax2.hist([success_lengths, failure_lengths], bins=20, alpha=0.7,
                label=['成功', '失败'], color=['green', 'red'])
        ax2.set_xlabel('轨迹长度（步数）')
        ax2.set_ylabel('频次')
        ax2.set_title('轨迹长度分布')
        ax2.legend()

        # 3. 轨迹示例 - 成功
        self._draw_environment(ax3)
        if success_trajectories:
            # 随机选择几个成功轨迹
            sample_success = np.random.choice(len(success_trajectories),
                                            min(5, len(success_trajectories)), replace=False)
            colors = cm.Greens(np.linspace(0.3, 1, len(sample_success)))

            for i, idx in enumerate(sample_success):
                traj = success_trajectories[idx]
                ax3.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.8)
                ax3.scatter(*traj[0], color=colors[i], s=50, marker='o')
                ax3.scatter(*traj[-1], color=colors[i], s=50, marker='X')

        ax3.set_title('成功轨迹示例')
        ax3.set_xlabel('X 坐标')
        ax3.set_ylabel('Y 坐标')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')

        # 4. 轨迹示例 - 失败
        self._draw_environment(ax4)
        if failure_trajectories:
            # 随机选择几个失败轨迹
            sample_failure = np.random.choice(len(failure_trajectories),
                                            min(5, len(failure_trajectories)), replace=False)
            colors = cm.Reds(np.linspace(0.3, 1, len(sample_failure)))

            for i, idx in enumerate(sample_failure):
                traj = failure_trajectories[idx]
                ax4.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, alpha=0.8)
                ax4.scatter(*traj[0], color=colors[i], s=50, marker='o')
                ax4.scatter(*traj[-1], color=colors[i], s=50, marker='x')

        ax4.set_title('失败轨迹示例')
        ax4.set_xlabel('X 坐标')
        ax4.set_ylabel('Y 坐标')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"成功失败分析图已保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def _draw_environment(self, ax):
        """绘制室内环境边界"""
        # 绘制墙壁边界
        wall_width = 0.5
        ax.add_patch(patches.Rectangle(
            (self.env_bounds['x_min'], self.env_bounds['y_min']),
            self.env_bounds['x_max'] - self.env_bounds['x_min'],
            self.env_bounds['y_max'] - self.env_bounds['y_min'],
            fill=False, edgecolor='black', linewidth=2, linestyle='--', alpha=0.7
        ))

        # 设置坐标轴范围
        ax.set_xlim(self.env_bounds['x_min'] - 1, self.env_bounds['x_max'] + 1)
        ax.set_ylim(self.env_bounds['y_min'] - 1, self.env_bounds['y_max'] + 1)

    def generate_summary_report(self, save_path: str = "agent/log/trajectory_analysis_report.txt"):
        """生成轨迹分析报告"""
        if not self.data:
            return

        summary = self.data['summary']

        report = f"""
轨迹分析报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

总体统计:
- 总回合数: {len(self.trajectories)}
- 成功回合: {summary['successful_episodes']}
- 碰撞回合: {summary['collision_episodes']}
- 成功率: {summary['successful_episodes']/len(self.trajectories)*100:.1f}%
- 平均奖励: {summary['average_reward']:.2f}
- 平均步数: {summary['average_steps']:.1f}

轨迹特征分析:
"""

        # 分析轨迹长度分布
        lengths = [len(traj['drone_trajectories'][0]) for traj in self.trajectories]
        report += f"- 轨迹长度范围: {min(lengths)} - {max(lengths)} 步\n"
        report += f"- 平均轨迹长度: {np.mean(lengths):.1f} 步\n"

        # 分析起点分布
        start_positions = [traj['start_position'] for traj in self.trajectories]
        start_x = [pos[0] for pos in start_positions]
        start_y = [pos[1] for pos in start_positions]
        report += f"- 起点X坐标范围: {min(start_x):.2f} - {max(start_x):.2f}\n"
        report += f"- 起点Y坐标范围: {min(start_y):.2f} - {max(start_y):.2f}\n"

        # 分析目标分布
        goal_positions = [traj['goal_position'] for traj in self.trajectories]
        goal_x = [pos[0] for pos in goal_positions]
        goal_y = [pos[1] for pos in goal_positions]
        report += f"- 目标X坐标范围: {min(goal_x):.2f} - {max(goal_x):.2f}\n"
        report += f"- 目标Y坐标范围: {min(goal_y):.2f} - {max(goal_y):.2f}\n"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"分析报告已保存: {save_path}")
        print(report)


def run_all_visualizations(trajectory_files: List[str] = None,
                          output_dir: str = 'muti_formation/agent/log/trajectory_visualizations'):
    """直接运行所有可视化功能"""
    if trajectory_files is None:
        # 自动查找所有轨迹文件
        base_dir = Path('muti_formation/agent/log')
        trajectory_files = []
        for file_path in base_dir.glob('trajectories_ep*.json'):
            trajectory_files.append(str(file_path))
        # 按episode编号排序
        trajectory_files.sort(key=lambda x: int(x.split('ep')[-1].split('.')[0]))
    
    # 创建可视化器
    visualizer = TrajectoryVisualizer(trajectory_files)

    if not visualizer.data:
        print("无法加载轨迹数据")
        return

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始生成轨迹可视化图，输出目录: {output_dir}")
    print(f"总回合数: {len(visualizer.trajectories)}")

    # 1. 生成轨迹热力图
    print("生成轨迹密度热力图...")
    heatmap_path = output_dir / "trajectory_heatmap_all.png"
    visualizer.plot_trajectory_heatmap(save_path=str(heatmap_path))

    # 2. 生成成功vs失败分析图
    print("生成成功失败分析图...")
    analysis_path = output_dir / "success_failure_analysis_all.png"
    visualizer.plot_success_vs_failure_analysis(str(analysis_path))

    # 3. 生成多轨迹叠加图（最后100个回合）
    print("生成多轨迹叠加图...")
    total_episodes = len(visualizer.trajectories)
    if total_episodes > 0:
        # 选择最后100个回合的episode_id
        num_to_select = min(100, total_episodes)
        episode_ids = [traj['episode_id'] for traj in visualizer.trajectories[-num_to_select:]]

        multi_path = output_dir / "multiple_trajectories.png"
        visualizer.plot_multiple_trajectories(episode_ids, num_to_select, str(multi_path))

    # 4. 生成分析报告
    print("生成轨迹分析报告...")
    report_path = output_dir / "trajectory_analysis_report_all.txt"
    visualizer.generate_summary_report(str(report_path))

    print("所有可视化图生成完成！")


def main():
    """主函数 - 处理命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(description='无人机轨迹可视化工具')
    parser.add_argument('--mode', choices=['all', 'analysis', 'report'],
                       default='all', help='运行模式')
    parser.add_argument('--trajectory_files', nargs='+', type=str,
                       help='轨迹数据文件路径列表，如果不指定则自动加载所有ep400-ep2000文件')
    parser.add_argument('--output_dir', type=str,
                       default='muti_formation/agent/log/trajectory_visualizations',
                       help='输出目录')
    parser.add_argument('--save_path', type=str, help='单个图的保存路径')

    args = parser.parse_args()

    if args.mode == 'all':
        run_all_visualizations(args.trajectory_files, args.output_dir)
    elif args.mode == 'analysis':
        visualizer = TrajectoryVisualizer(args.trajectory_files or ['muti_formation/agent/log/leader_phase1_final_trajectories.json'])
        if visualizer.data:
            visualizer.plot_success_vs_failure_analysis(args.save_path)
    elif args.mode == 'report':
        visualizer = TrajectoryVisualizer(args.trajectory_files or ['muti_formation/agent/log/leader_phase1_final_trajectories.json'])
        if visualizer.data:
            visualizer.generate_summary_report(args.save_path or "agent/log/trajectory_analysis_report.txt")


if __name__ == '__main__':
    # 检查命令行参数
    if len(sys.argv) == 1:
        # 没有参数，直接运行所有可视化（加载所有轨迹文件）
        run_all_visualizations()
    else:
        # 有参数，使用命令行模式
        main()