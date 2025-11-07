import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from config import observation_space_v1,observation_space_v0

# 可修改此变量为 'v0' 或 'v1' 来切换不同版本
version = 'v0'
observation_space_v = eval(f"observation_space_{version}")

def plot_3d_range():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制无人机活动范围
    x1 = [observation_space_v["drone_lower_bound_x"], observation_space_v["drone_upper_bound_x"]]
    y1 = [observation_space_v["drone_lower_bound_y"], observation_space_v["drone_upper_bound_y"]]
    z1 = [observation_space_v["drone_lower_bound_z"], observation_space_v["drone_upper_bound_z"]]

    xx1, yy1 = np.meshgrid(x1, y1)
    ax.plot_wireframe(xx1, yy1, np.full_like(xx1, z1[0]), color='b', alpha=0.3, label='Drone Range')
    ax.plot_wireframe(xx1, yy1, np.full_like(xx1, z1[1]), color='b', alpha=0.3)

    xx1, zz1 = np.meshgrid(x1, z1)
    ax.plot_wireframe(xx1, np.full_like(xx1, y1[0]), zz1, color='b', alpha=0.3)
    ax.plot_wireframe(xx1, np.full_like(xx1, y1[1]), zz1, color='b', alpha=0.3)

    yy1, zz1 = np.meshgrid(y1, z1)
    ax.plot_wireframe(np.full_like(yy1, x1[0]), yy1, zz1, color='b', alpha=0.3)
    ax.plot_wireframe(np.full_like(yy1, x1[1]), yy1, zz1, color='b', alpha=0.3)

    # 绘制目标活动范围
    x2 = [observation_space_v["goal_lower_bound_x"], observation_space_v["goal_upper_bound_x"]]
    y2 = [observation_space_v["goal_lower_bound_y"], observation_space_v["goal_upper_bound_y"]]
    z2 = [observation_space_v["goal_lower_bound_z"], observation_space_v["goal_upper_bound_z"]]

    xx2, yy2 = np.meshgrid(x2, y2)
    ax.plot_wireframe(xx2, yy2, np.full_like(xx2, z2[0]), color='r', alpha=0.3, label='Goal Range')
    ax.plot_wireframe(xx2, yy2, np.full_like(xx2, z2[1]), color='r', alpha=0.3)

    xx2, zz2 = np.meshgrid(x2, z2)
    ax.plot_wireframe(xx2, np.full_like(xx2, y2[0]), zz2, color='r', alpha=0.3)
    ax.plot_wireframe(xx2, np.full_like(xx2, y2[1]), zz2, color='r', alpha=0.3)

    yy2, zz2 = np.meshgrid(y2, z2)
    ax.plot_wireframe(np.full_like(yy2, x2[0]), yy2, zz2, color='r', alpha=0.3)
    ax.plot_wireframe(np.full_like(yy2, x2[1]), yy2, zz2, color='r', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Drone and Goal Activity Range in observation_space_{version}')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    plot_3d_range()
