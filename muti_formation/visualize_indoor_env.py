
"""
室内环境可视化脚本 - 只显示环境，不包含无人机
"""
import sys
import os
sys.path.append('.')

from drone_envs.envs.drone_env_multi import DroneNavigationMulti
import pybullet as p
import time

def visualize_indoor_environment():
    """可视化室内环境布局"""
    print("🏠 室内环境可视化 - 只显示环境布局")
    print("=" * 50)

    try:
        # 创建室内环境（不包含无人机）
        env = DroneNavigationMulti(
            num_drones=5,  # 虽然不显示无人机，但需要保持配置一致
            environment_type="indoor",
            use_depth_camera=False
        )

        print("✅ 室内环境创建成功！")
        print("\n📊 环境统计信息:")
        print(f"   房间尺寸: {env.environment_manager.indoor_config['room_size']} x {env.environment_manager.indoor_config['room_size']} x {env.environment_manager.indoor_config['wall_height']} 米")
        print(f"   外墙厚度: {env.environment_manager.indoor_config['wall_thickness']} 米")
        print(f"   总障碍物数量: {len(env.environment_manager.walls) + len(env.environment_manager.obstacles)}")
        print(f"   外墙数量: {len(env.environment_manager.walls)} 个")
        print(f"   圆柱体障碍物: {len(env.environment_manager.obstacles)} 个")
        print(f"   起点基准位置: 左下角 (-{env.environment_manager.indoor_config['room_size']/2 - 2:.1f}, -{env.environment_manager.indoor_config['room_size']/2 - 2:.1f})")
        print(f"   终点固定位置: 右上角 ({env.environment_manager.indoor_config['room_size']/2 - 2:.1f}, {env.environment_manager.indoor_config['room_size']/2 - 2:.1f})")

        # 设置最佳观察视角
        p.resetDebugVisualizerCamera(
            cameraDistance=30.0,  # 稍微拉远一点
            cameraYaw=45,         # 45度角
            cameraPitch=-35,      # 稍微向下看
            cameraTargetPosition=[0, 0, 1.5]  # 看向房间中央
        )

        print("\n🎨 环境布局说明:")
        print("   🟦 白色区域 = 外墙包围的房间空间")
        print("    灰色柱子 = 圆柱体障碍物")
        print("   🎯 绿色球体 = 目标位置（右上角）")
        print("   🚁 无人机编队 = 起点位置（左下角）")

        print("\n🎮 控制说明:")
        print("   🖱️  鼠标左键拖拽: 旋转视角")
        print("   🖱️  鼠标右键拖拽: 平移视角")
        print("   🖱️  鼠标滚轮: 缩放")
        print("   ⌨️  按 'Ctrl+C' 退出")

        print("\n🏗️  环境组成:")
        print("   • 外墙: 4面 (北、南、东、西)")
        print("   • 内墙: 4面 (创建中央走廊)")
        print("   • 圆柱体: 25个 (直径0.6m, 高度2m, 灰色)")
        print("   • 起点: 左下角编队位置")
        print("   • 终点: 右上角固定位置")

        # 让用户观察静态环境
        print("\n⏸️  现在您可以仔细观察室内环境布局...")
        print("   环境已加载完成，请查看各个组件的位置和布局")
        print("   注意：起点在左下角，终点在右上角")

        # 等待用户观察
        input("\n按Enter键开始无人机位置演示...")

        print("\n� 开始无人机位置演示...")

        # 显示无人机起始位置
        print("\n📍 无人机编队起始位置:")
        for i, drone in enumerate(env.drones):
            pos, _ = p.getBasePositionAndOrientation(drone.drone, env.client)
            print(f"   无人机{i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        print(f"\n🎯 目标位置: ({env.goal[0]:.2f}, {env.goal[1]:.2f}, {env.goal[2]:.2f})")

        # 计算距离
        leader_pos, _ = p.getBasePositionAndOrientation(env.drones[0].drone, env.client)
        distance = ((env.goal[0] - leader_pos[0])**2 + (env.goal[1] - leader_pos[1])**2)**0.5
        print(f"📏 起点到终点距离: {distance:.2f} 米")

        print("\n✅ 演示完成！")
        print("💡 提示: 您可以继续在GUI中观察，或按Enter键退出")

        input("\n按Enter键退出...")

        env.close()
        print("👋 感谢观察室内环境！")

    except Exception as e:
        print(f"\n❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()

def show_environment_details():
    """显示环境详细信息"""
    print("🔍 室内环境详细说明:")
    print("=" * 60)
    print("📐 物理布局:")
    print("   • 总面积: 30m x 30m")
    print("   • 房间高度: 3m")
    print("   • 外墙厚度: 0.2m")
    print()
    print("🚧 障碍物分布:")
    print("   • 外墙: 4面 (北、南、东、西)")
    print("   • 内墙: 4面 (创建中央走廊)")
    print("   • 圆柱体: 25个 (直径0.6m, 高度2m, 灰色)")
    print()
    print("🎯 导航特点:")
    print("   • 起点: 左下角编队 (-14,-14)")
    print("   • 终点: 右上角固定 (13,13)")
    print("   • 挑战: 密集障碍物 + 编队控制")
    print("   • 适合: 多无人机编队避障研究")
    print("=" * 60)

if __name__ == "__main__":
    print("🎯 PyBullet 室内环境可视化")
    print("=" * 60)

    show_environment_details()

    try:
        visualize_indoor_environment()
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")

    print("\n🎉 可视化结束！")
