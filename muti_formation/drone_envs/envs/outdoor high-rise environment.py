import pybullet as p
import random
import numpy as np
from typing import Tuple
import pybullet_data
from .drone_env_multi import DroneNavigationMulti
from ..resources.drone import Drone
from ..resources.plane import Plane
from ..resources.goal import Goal
from ..resources.building import Building

def _spawn_simple_obstacles(self):
        """生成复杂城市高楼环境"""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        Plane(self.client)
        
        # 初始化障碍物列表
        self.obstacles_pos_list = []
        self.obstacle_ids = []
        self.all_buildings = []  # 存储所有高楼信息
        
        # 固定起点高楼（比较矮，从地面开始）
        start_building_height = 2.0
        start_building_pos = [0, 0, start_building_height / 2.0 + 0.1]  # 从地面开始，稍微高一点避免与地面重合
        start_building = Building(self.client, start_building_pos, start_building_height)
        self.obstacle_ids.append(start_building.id)
        self.obstacles_pos_list.append([0, 0, start_building_height / 2.0])  # 存储实际位置
        self.all_buildings.append([0, 0, start_building_height])  # 存储高楼信息
        
        # 生成复杂城市高楼：稀疏网格布局，高楼从地面开始
        grid_x = np.arange(-35, 36, 6)  # x 坐标：-35 到 35，步长 6，更稀疏
        grid_y = np.arange(-35, 36, 6)  # y 坐标：-35 到 35，步长 6
        for x in grid_x:
            for y in grid_y:
                dist_from_origin = np.sqrt(x**2 + y**2)
                if dist_from_origin < 3:  # 避开起点附近
                    continue
                if random.random() < 0.5:  # 50% 概率放置高楼，降低密度
                    # 随机高度：1 到 12 层，高楼更显著
                    height_layers = random.choices([1,2,3,4,5,6,8,10,12], weights=[0.15,0.15,0.15,0.15,0.15,0.1,0.08,0.05,0.02])[0]
                    height = height_layers * 1.0
                    building = Building(self.client, [x + random.uniform(-1.5, 1.5), y + random.uniform(-1.5, 1.5), height / 2.0 + 0.1], height)
                    self.obstacle_ids.append(building.id)
                    self.obstacles_pos_list.append([x, y, height / 2.0])  # 存储实际位置
                    self.all_buildings.append([x, y, height])  # 存储高楼信息
        
        # 添加少量随机分散的高楼，降低复杂性
        for _ in range(25):  # 从50减少到25
            x = random.uniform(-40, 40)
            y = random.uniform(-40, 40)
            dist_from_origin = np.sqrt(x**2 + y**2)
            if dist_from_origin < 5:  # 避开起点
                continue
            height_layers = random.randint(2, 10)  # 从1-5层改为2-10层，让高楼更高
            height = height_layers * 1.0
            building = Building(self.client, [x + random.uniform(-1, 1), y + random.uniform(-1, 1), height / 2.0 + 0.1], height)
            self.obstacle_ids.append(building.id)
            self.obstacles_pos_list.append([x, y, height / 2.0])
            self.all_buildings.append([x, y, height])  # 存储高楼信息
        

def _sample_goal(self) -> Tuple[float, float, float]:
        """从现有高楼中随机选择一个作为终点"""
        if hasattr(self, 'all_buildings') and self.all_buildings:
            # 过滤出距离起点足够远的高楼
            valid_buildings = []
            for building in self.all_buildings:
                x, y, height = building
                dist_from_start = np.sqrt(x**2 + y**2)
                if dist_from_start > 15:  # 距离起点至少15米
                    valid_buildings.append(building)
            
            if valid_buildings:
                # 使用环境的随机数生成器进行随机选择
                goal_building = random.choice(valid_buildings)
                goal_x, goal_y, goal_height = goal_building
                goal_z = goal_height + random.uniform(2, 4)  # 在高楼上方2-4米
                return goal_x, goal_y, goal_z
        
        # 备用方案，如果没有合适的高楼
        return (
            float(random.uniform(20, 35)),
            float(random.uniform(20, 35)),
            float(random.uniform(5, 12))
        )