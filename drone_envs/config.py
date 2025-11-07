import pybullet as p

# 定义无人机环境 v0 的配置参数    V0没有障碍物
drone_env_v0 = {
    # 使用 PyBullet 的图形用户界面模式显示仿真环境
    "display": p.GUI,
    # X 轴方向推力的下限
    "thrust_x_lower_bound": -0.1,
    # Y 轴方向推力的下限
    "thrust_y_lower_bound": -0.1,
    # Z 轴方向推力的下限
    "thrust_z_lower_bound": -0.1,
    # X 轴方向推力的上限
    "thrust_x_upper_bound": 0.1,
    # Y 轴方向推力的上限
    "thrust_y_upper_bound": 0.1,
    # Z 轴方向推力的上限
    "thrust_z_upper_bound": 0.1,
    # 无人机观测空间的维度
    "drone_observation_space": 9,
}

# 定义无人机环境 v0 的观测空间配置参数
observation_space_v0 = {
    # 无人机在 X 轴方向位置的下限
    "drone_lower_bound_x": -12, 
    # 无人机在 X 轴方向位置的上限
    "drone_upper_bound_x": 12,     
    # 无人机在 Y 轴方向位置的下限
    "drone_lower_bound_y": -12, 
    # 无人机在 Y 轴方向位置的上限
    "drone_upper_bound_y": 12, 
    # 无人机在 Z 轴方向位置的下限
    "drone_lower_bound_z": 0, 
    # 无人机在 Z 轴方向位置的上限
    "drone_upper_bound_z": 12, 
    # 无人机在 X 轴方向速度的下限
    "drone_velocity_lower_bound_x": -3, 
    # 无人机在 X 轴方向速度的上限
    "drone_velocity_upper_bound_x": 3,     
    # 无人机在 Y 轴方向速度的下限
    "drone_velocity_lower_bound_y": -3, 
    # 无人机在 Y 轴方向速度的上限
    "drone_velocity_upper_bound_y": 3, 
    # 无人机在 Z 轴方向速度的下限
    "drone_velocity_lower_bound_z": -3, 
    # 无人机在 Z 轴方向速度的上限
    "drone_velocity_upper_bound_z": 3,     
    # 目标在 X 轴方向位置的下限
    "goal_lower_bound_x": -9,
    # 目标在 X 轴方向位置的上限
    "goal_upper_bound_x": 9,
    # 目标在 Y 轴方向位置的下限
    "goal_lower_bound_y": -9,
    # 目标在 Y 轴方向位置的上限
    "goal_upper_bound_y": 9,
    # 目标在 Z 轴方向位置的下限
    "goal_lower_bound_z": 0,
    # 目标在 Z 轴方向位置的上限
    "goal_upper_bound_z": 9,
}

# 定义无人机环境 v1 的配置参数  V1有障碍物
drone_env_v1 = {
    # 使用 PyBullet 的图形用户界面模式显示仿真环境
    "display": p.GUI,
    # X 轴方向推力的下限
    "thrust_x_lower_bound": -0.2,
    # Y 轴方向推力的下限
    "thrust_y_lower_bound": -0.2,
    # Z 轴方向推力的下限
    "thrust_z_lower_bound": -0.2,
    # X 轴方向推力的上限
    "thrust_x_upper_bound": 0.2,
    # Y 轴方向推力的上限
    "thrust_y_upper_bound": 0.2,
    # Z 轴方向推力的上限
    "thrust_z_upper_bound": 0.2,
    # 相机的像素大小
    "camera_pixel": 16,
    # 无人机元数据空间的维度
    "drone_metadata_space": 9,
}

# 定义无人机环境 v1 的观测空间配置参数
observation_space_v1 = {
    # 无人机在 X 轴方向位置的下限
    "drone_lower_bound_x": -25, 
    # 无人机在 X 轴方向位置的上限
    "drone_upper_bound_x": 25,     
    # 无人机在 Y 轴方向位置的下限
    "drone_lower_bound_y": -25, 
    # 无人机在 Y 轴方向位置的上限
    "drone_upper_bound_y": 25, 
    # 无人机在 Z 轴方向位置的下限
    "drone_lower_bound_z": 0, 
    # 无人机在 Z 轴方向位置的上限
    "drone_upper_bound_z": 15, 
    # 无人机在 X 轴方向速度的下限
    "drone_velocity_lower_bound_x": -3, 
    # 无人机在 X 轴方向速度的上限
    "drone_velocity_upper_bound_x": 3,     
    # 无人机在 Y 轴方向速度的下限
    "drone_velocity_lower_bound_y": -3, 
    # 无人机在 Y 轴方向速度的上限
    "drone_velocity_upper_bound_y": 3, 
    # 无人机在 Z 轴方向速度的下限
    "drone_velocity_lower_bound_z": -3, 
    # 无人机在 Z 轴方向速度的上限
    "drone_velocity_upper_bound_z": 3,     
    # 目标在 X 轴方向位置的下限
    "goal_lower_bound_x": -21,
    # 目标在 X 轴方向位置的上限
    "goal_upper_bound_x": 21,
    # 目标在 Y 轴方向位置的下限
    "goal_lower_bound_y": -21,
    # 目标在 Y 轴方向位置的上限
    "goal_upper_bound_y": 21,
    # 目标在 Z 轴方向位置的下限
    "goal_lower_bound_z": 0,
    # 目标在 Z 轴方向位置的上限
    "goal_upper_bound_z": 12,
}