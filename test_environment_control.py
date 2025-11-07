"""
使用环境真实控制逻辑测试无人机物理行为
- 平面模式: enforce_planar=True
- 不提供重力补偿(因为重力设为0)
- 每步后强制约束姿态和Z速度
"""

import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from muti_formation.drone_envs.resources.drone import Drone
from muti_formation.drone_envs.config import multi_drone_env as config

print("="*80)
print("使用环境真实控制逻辑测试无人机")
print("="*80)

save_dir = "drone_control_analysis_v2"
os.makedirs(save_dir, exist_ok=True)

def test_with_environment_logic():
    """测试: 使用环境中的真实控制逻辑"""
    print("\n" + "="*80)
    print("测试: 模拟环境中的真实控制逻辑")
    print("="*80)
    
    # 创建物理仿真
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 🔥 关键: 平面模式下重力为0 (从环境中复制的逻辑)
    print("\n重力设置: 0 (平面模式)")
    p.setGravity(0, 0, 0, physicsClientId=client)  # 平面模式: 重力为0
    
    # 设置物理参数(从环境复制)
    dt = 1/30.0  # 环境使用的时间步长
    p.setPhysicsEngineParameter(
        fixedTimeStep=dt,
        numSolverIterations=50,
        numSubSteps=8,
        enableConeFriction=1,
        restitutionVelocityThreshold=0.05,
        contactERP=0.8,
        frictionERP=0.8,
        physicsClientId=client
    )
    
    # 加载地面
    p.loadURDF("plane.urdf", physicsClientId=client)
    
    # 创建无人机
    drone = Drone(client)
    drone_id = drone.drone
    
    # 获取无人机质量(用于可能的重力补偿)
    drone_mass = p.getDynamicsInfo(drone_id, -1, physicsClientId=client)[0]
    print(f"无人机质量: {drone_mass} kg")
    
    # 重置到起始位置,高度1.6米
    start_height = config.get('start_height', 1.6)
    p.resetBasePositionAndOrientation(
        drone_id, 
        [0, 0, start_height],
        p.getQuaternionFromEuler([0, 0, 0]),
        physicsClientId=client
    )
    
    # 数据记录
    positions = []
    velocities = []
    orientations = []
    time_steps = []
    
    # 测试参数
    test_duration = 5.0
    thrust_value = 0  # 测试thrust
    torque_value = -0.1   # 测试torque
    
    print(f"\n测试配置:")
    print(f"  - enforce_planar: True")
    print(f"  - 重力: 0 (平面模式)")
    print(f"  - 测试时长: {test_duration}秒")
    print(f"  - Thrust值: {thrust_value}")
    print(f"  - Torque值: {torque_value}")
    print(f"  - 初始位置: [0, 0, {start_height}]")
    print(f"  - 时间步长: {dt}秒")
    
    num_steps = int(test_duration / dt)
    
    print(f"\n开始仿真... (共{num_steps}步)")
    
    for step in range(num_steps):
        # 1. 施加动作(从环境复制的逻辑)
        action = np.array([thrust_value, torque_value])
        
        # drone.apply_action() - 不提供重力补偿
        drone.apply_action(action, apply_gravity_compensation=False)
        
        # 2. 环境中的重力补偿逻辑
        # 3D模式才补偿,平面模式不补偿
        enforce_planar = True
        if not enforce_planar:
            gravity_compensation = drone_mass * 9.8
            p.applyExternalForce(drone_id, -1, [0, 0, gravity_compensation], 
                               [0, 0, 0], p.WORLD_FRAME, physicsClientId=client)
        # else: 平面模式不提供重力补偿
        
        # 3. 物理仿真
        p.stepSimulation(physicsClientId=client)
        
        # 4. 平面模式约束(从环境复制)
        if enforce_planar:
            # 获取当前状态
            current_pos, current_orn = p.getBasePositionAndOrientation(drone_id, physicsClientId=client)
            current_vel, current_ang_vel = p.getBaseVelocity(drone_id, physicsClientId=client)
            current_euler = p.getEulerFromQuaternion(current_orn)
            
            # 【关键】保留xy速度,强制z速度为0
            constrained_vel = [current_vel[0], current_vel[1], 0.0]
            
            # 强制姿态为水平
            constrained_euler = [0.0, 0.0, current_euler[2]]  # 只保留yaw
            constrained_orn = p.getQuaternionFromEuler(constrained_euler)
            
            # 强制z位置为固定高度
            constrained_pos = [current_pos[0], current_pos[1], start_height]
            
            # 重置姿态和位置
            p.resetBasePositionAndOrientation(drone_id, constrained_pos, constrained_orn, 
                                            physicsClientId=client)
            
            # 重置速度
            constrained_ang_vel = [0.0, 0.0, current_ang_vel[2]]
            p.resetBaseVelocity(drone_id, constrained_vel, constrained_ang_vel, 
                              physicsClientId=client)
        
        # 5. 记录数据
        pos, orn = p.getBasePositionAndOrientation(drone_id, physicsClientId=client)
        vel, ang_vel = p.getBaseVelocity(drone_id, physicsClientId=client)
        euler = p.getEulerFromQuaternion(orn)
        
        positions.append(list(pos))
        velocities.append(list(vel))
        orientations.append(list(euler))
        time_steps.append(step * dt)
        
        if step % 30 == 0:
            print(f"  步骤 {step}/{num_steps}: pos={pos}, vel={vel}, yaw={np.degrees(euler[2]):.1f}°")
    
    p.disconnect(physicsClientId=client)
    
    # 转换为numpy数组
    positions = np.array(positions)
    velocities = np.array(velocities)
    orientations = np.array(orientations)
    time_steps = np.array(time_steps)
    
    # 分析结果
    print(f"\n" + "-"*80)
    print("测试结果分析:")
    print("-"*80)
    
    start_pos = positions[0]
    end_pos = positions[-1]
    displacement = end_pos - start_pos
    total_distance = np.linalg.norm(displacement[:2])
    
    print(f"\n位置变化:")
    print(f"  - 起始位置: {start_pos}")
    print(f"  - 结束位置: {end_pos}")
    print(f"  - 水平移动距离: {total_distance:.4f} 米")
    print(f"  - X位移: {displacement[0]:.4f} 米")
    print(f"  - Y位移: {displacement[1]:.4f} 米")
    print(f"  - Z位移: {displacement[2]:.4f} 米")
    
    # 检查高度是否保持
    z_positions = positions[:, 2]
    z_min = np.min(z_positions)
    z_max = np.max(z_positions)
    z_std = np.std(z_positions)
    
    print(f"\n高度控制:")
    print(f"  - 目标高度: {start_height} 米")
    print(f"  - Z最小值: {z_min:.6f} 米")
    print(f"  - Z最大值: {z_max:.6f} 米")
    print(f"  - Z标准差: {z_std:.6f} 米")
    
    if z_std < 0.001:
        print(f"  ✅ 高度保持完美! 标准差 < 0.001米")
    elif z_std < 0.01:
        print(f"  ✅ 高度保持良好! 标准差 < 0.01米")
    else:
        print(f"  ⚠️ 高度有波动! 标准差 = {z_std:.6f}米")
    
    print(f"\n速度:")
    end_vel = velocities[-1]
    max_vel_xy = np.max(np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2))
    max_vel_z = np.max(np.abs(velocities[:, 2]))
    
    print(f"  - 结束速度: {end_vel}")
    print(f"  - 最大XY速度: {max_vel_xy:.4f} m/s")
    print(f"  - 最大Z速度: {max_vel_z:.6f} m/s")
    
    if max_vel_z < 0.001:
        print(f"  ✅ Z速度完全约束! < 0.001 m/s")
    
    print(f"\n偏航角:")
    start_yaw = np.degrees(orientations[0, 2])
    end_yaw = np.degrees(orientations[-1, 2])
    yaw_change = end_yaw - start_yaw
    
    print(f"  - 起始偏航: {start_yaw:.2f}°")
    print(f"  - 结束偏航: {end_yaw:.2f}°")
    print(f"  - 偏航变化: {yaw_change:.2f}°")
    
    # 检查Roll和Pitch
    max_roll = np.max(np.abs(orientations[:, 0]))
    max_pitch = np.max(np.abs(orientations[:, 1]))
    
    print(f"\n姿态约束:")
    print(f"  - 最大Roll角: {np.degrees(max_roll):.6f}°")
    print(f"  - 最大Pitch角: {np.degrees(max_pitch):.6f}°")
    
    if max_roll < 0.001 and max_pitch < 0.001:
        print(f"  ✅ 姿态完全水平! Roll和Pitch < 0.001°")
    
    # 判断结果
    print(f"\n" + "="*80)
    print("总体评估:")
    print("="*80)
    
    if total_distance > 0.1:
        print(f"✅ Thrust控制有效! 移动了{total_distance:.4f}米")
    else:
        print(f"❌ Thrust控制无效! 几乎没有移动")
    
    if abs(yaw_change) > 1.0:
        print(f"✅ Torque控制有效! 旋转了{yaw_change:.2f}°")
    else:
        print(f"⚠️ Torque控制较弱或torque值太小")
    
    if z_std < 0.01:
        print(f"✅ 高度保持完美! 标准差{z_std:.6f}米")
    else:
        print(f"❌ 高度保持失败!")
    
    if max_vel_z < 0.001:
        print(f"✅ Z速度约束完美! < 0.001 m/s")
    
    if max_roll < 0.001 and max_pitch < 0.001:
        print(f"✅ 姿态约束完美! Roll/Pitch < 0.001°")
    
    print("="*80)
    
    # 绘图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Environment Control Logic Test\n(thrust={thrust_value}, torque={torque_value}, planar mode)', 
                 fontsize=14, weight='bold')
    
    # 1. XY轨迹
    axes[0, 0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Path')
    axes[0, 0].scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='Start', zorder=5)
    axes[0, 0].scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, label='End', zorder=5)
    
    # 绘制朝向箭头
    for i in range(0, len(positions), 15):
        yaw = orientations[i, 2]
        dx = 0.5 * np.cos(yaw)
        dy = 0.5 * np.sin(yaw)
        axes[0, 0].arrow(positions[i, 0], positions[i, 1], dx, dy,
                        head_width=0.15, head_length=0.1, fc='orange', ec='orange', alpha=0.6)
    
    axes[0, 0].set_xlabel('X (m)', fontsize=12)
    axes[0, 0].set_ylabel('Y (m)', fontsize=12)
    axes[0, 0].set_title('XY Trajectory with Heading', fontsize=12, weight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # 2. 位置vs时间
    axes[0, 1].plot(time_steps, positions[:, 0], label='X', linewidth=2)
    axes[0, 1].plot(time_steps, positions[:, 1], label='Y', linewidth=2)
    axes[0, 1].plot(time_steps, positions[:, 2], label='Z', linewidth=2, linestyle='--')
    axes[0, 1].axhline(y=start_height, color='r', linestyle=':', alpha=0.5, label=f'Target Z={start_height}')
    axes[0, 1].set_xlabel('Time (s)', fontsize=12)
    axes[0, 1].set_ylabel('Position (m)', fontsize=12)
    axes[0, 1].set_title('Position vs Time', fontsize=12, weight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Z位置(放大)
    axes[1, 0].plot(time_steps, positions[:, 2], 'b-', linewidth=2)
    axes[1, 0].axhline(y=start_height, color='r', linestyle='--', label=f'Target={start_height}')
    axes[1, 0].set_xlabel('Time (s)', fontsize=12)
    axes[1, 0].set_ylabel('Z Position (m)', fontsize=12)
    axes[1, 0].set_title(f'Height Control (std={z_std:.6f}m)', fontsize=12, weight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 速度
    axes[1, 1].plot(time_steps, velocities[:, 0], label='Vx', linewidth=2)
    axes[1, 1].plot(time_steps, velocities[:, 1], label='Vy', linewidth=2)
    axes[1, 1].plot(time_steps, velocities[:, 2]*1000, label='Vz×1000', linewidth=2, linestyle='--')
    axes[1, 1].set_xlabel('Time (s)', fontsize=12)
    axes[1, 1].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[1, 1].set_title('Velocity vs Time', fontsize=12, weight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. 偏航角
    axes[2, 0].plot(time_steps, np.degrees(orientations[:, 2]), 'b-', linewidth=2)
    axes[2, 0].set_xlabel('Time (s)', fontsize=12)
    axes[2, 0].set_ylabel('Yaw Angle (degrees)', fontsize=12)
    axes[2, 0].set_title(f'Yaw Angle (Δ={yaw_change:.2f}°)', fontsize=12, weight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Roll和Pitch(应该为0)
    axes[2, 1].plot(time_steps, np.degrees(orientations[:, 0])*1000, label='Roll×1000', linewidth=2)
    axes[2, 1].plot(time_steps, np.degrees(orientations[:, 1])*1000, label='Pitch×1000', linewidth=2)
    axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2, 1].set_xlabel('Time (s)', fontsize=12)
    axes[2, 1].set_ylabel('Angle (degrees ×1000)', fontsize=12)
    axes[2, 1].set_title('Roll/Pitch Constraint (should be 0)', fontsize=12, weight='bold')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'environment_control_test.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {save_path}")
    plt.close()
    
    return {
        'total_distance': total_distance,
        'yaw_change': yaw_change,
        'z_std': z_std,
        'max_vel_z': max_vel_z
    }


# 运行测试
print("\n" + "🚁"*40)
print("开始环境控制逻辑测试")
print("🚁"*40)

result = test_with_environment_logic()

print("\n" + "="*80)
print("测试完成!")
print("="*80)
print(f"\n关键指标:")
print(f"  - 水平移动: {result['total_distance']:.4f} 米")
print(f"  - 偏航变化: {result['yaw_change']:.2f} 度")
print(f"  - 高度标准差: {result['z_std']:.6f} 米")
print(f"  - 最大Z速度: {result['max_vel_z']:.6f} m/s")
print(f"\n结果保存在: {save_dir}/")
print("="*80)
