"""
环境设置和场景生成模块
"""
import numpy as np
import pybullet as p
from typing import Dict, Any, List, Tuple
from ..resources.plane import Plane
from ..resources.goal import Goal
from ..resources.indoor_components import IndoorWall, IndoorBox, IndoorCylinder, IndoorOuterWall
from ..config import multi_drone_env as config
import random

class EnvironmentManager:
    """环境管理器 - 负责场景生成、目标采样等"""
    
    def __init__(self, client: int, config: Dict[str, Any]):
        """
        初始化环境管理器
        
        Args:
            client: PyBullet客户端ID
            config: 环境配置参数
        """
        self.client = client
        self.config = config
        self.environment_type = config.get('environment_type', 'indoor')  # 恢复：默认使用室内环境
        
        
        # 室内环境配置 
        self.indoor_config = {
            'room_size': config.get('room_size', 30),  # 修正：使用30米房间
            'wall_height': config.get('wall_height', 3.0),
            'wall_thickness': config.get('wall_thickness', 0.2),
            'obstacle_count': config.get('obstacle_count', 5),  # 修复：从10进一步降低到5，仅7个障碍物(5柱+2墙)
            'person_count': config.get('person_count', 0)  # 移除person
        }
        
        # 目标采样配置
        self.goal_config = {
            'min_distance': config.get('min_goal_distance', 8.0),
            'max_distance': config.get('max_goal_distance', 15.0),
            'goal_height': config.get('goal_height', 1.5),  # 恢复，使用config传递的值
            'safe_margin': config.get('goal_safe_margin', 1.0)
        }
        
        # 起始位置配置
        self.start_config = {
            'formation_spacing': config.get('formation_spacing', 1.0),
            'start_height': config.get('start_height', 1.5),  # 恢复，使用config传递的值
            'safe_zone_radius': config.get('safe_zone_radius', 3.0)
        }
        
        # 存储生成的对象
        self.walls = []
        self.obstacles = []
        self.persons = []
        self.goal_id = None
        
    def setup_physics_world(self, dt: float = 1/30, enforce_planar: bool = False):
        """设置物理世界参数"""
        p.setTimeStep(dt, self.client)
        
        # 对于平面模式，设置重力为0，让无人机完全控制自己的运动
        if enforce_planar:
            p.setGravity(0, 0, 0, physicsClientId=self.client)
        else:
            # 3D模式下保持正常重力
            p.setGravity(0, 0, -9.8, physicsClientId=self.client)
            
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        
        # 添加地面
        ground_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        ground_id = p.createMultiBody(0, ground_shape, physicsClientId=self.client)
        p.changeVisualShape(ground_id, -1, rgbaColor=[0.8, 0.8, 0.8, 1.0], physicsClientId=self.client)
        
    def generate_environment(self) -> Dict[str, Any]:
        """生成环境场景"""
        if self.environment_type == "indoor":
            return self._generate_indoor_environment()
        else:
            return self._generate_simple_environment()
    
    def _generate_indoor_environment(self) -> Dict[str, Any]:
        """生成30m x 30m的室内环境 - 外墙 + 4面内墙 + 圆柱障碍"""
        room_size = self.indoor_config['room_size']
        wall_height = self.indoor_config['wall_height']
        wall_thickness = self.indoor_config['wall_thickness']
        
        # 清理旧对象
        self._clear_environment()
        
        # 生成外墙 (30m x 30m)
        self._generate_outer_walls(room_size, wall_height, wall_thickness)
        
        # 生成4面内部墙壁
        self._generate_four_interior_walls(room_size, wall_height)
        
        # 生成15个圆柱障碍物
        self._generate_cylinder_obstacles()
        
        return {
            'type': 'indoor',
            'room_size': room_size,
            'wall_count': len(self.walls),
            'obstacle_count': len(self.obstacles),
            'person_count': 0  # 删除person
        }
    
    def _generate_simple_environment(self) -> Dict[str, Any]:
        """生成简单的户外环境"""
        # 清理旧对象
        self._clear_environment()
        
        # 生成简单平面
        plane = Plane(self.client)
        
        return {
            'type': 'simple',
            'objects': ['ground_plane']
        }
    
    def _generate_outer_walls(self, room_size: float, wall_height: float, wall_thickness: float):
        """生成完整的四面外墙围成30m x 30m正方形"""
        half_size = room_size / 2
        
        # 四面外墙配置 - 确保完整围成正方形
        wall_configs = [
            # 北墙 (上边) - 水平放置
            {'position': [0, half_size, wall_height/2], 'orientation': [0, 0, 0, 1]},
            # 南墙 (下边) - 水平放置 
            {'position': [0, -half_size, wall_height/2], 'orientation': [0, 0, 0, 1]},
            # 东墙 (右边) - 垂直放置，需要旋转90度
            {'position': [half_size, 0, wall_height/2], 'orientation': [0, 0, 0.7071, 0.7071]},
            # 西墙 (左边) - 垂直放置，需要旋转90度
            {'position': [-half_size, 0, wall_height/2], 'orientation': [0, 0, 0.7071, 0.7071]}
        ]
        
        for config in wall_configs:
            wall = IndoorOuterWall(
                self.client, 
                position=config['position'],
                orientation=config['orientation']
            )
            self.walls.append(wall)
    
    def _generate_four_interior_walls(self, room_size: float, wall_height: float):
        """生成4面独立的内部墙壁，分散分布，不相连"""
        half_size = room_size / 2
        
        # 4面独立的内部墙壁配置 - 分散布置避免交叉
        interior_walls = [
            # 右上区域 - 水平墙，往左挪一点
            {'pos': [half_size * 0.5, half_size * 0.5, wall_height/2], 'ori': 0},
            
            # 左下区域 - 水平墙 
            {'pos': [-half_size * 0.5, -half_size * 0.5, wall_height/2], 'ori': 0},

            # 右下区域 - 垂直墙
            {'pos': [half_size * 0.5, -half_size * 0.5, wall_height/2], 'ori': np.pi/2},
            
            # 左上区域 - 垂直墙
            {'pos': [-half_size * 0.5, half_size * 0.5, wall_height/2], 'ori': np.pi/2}
        ]
        
        for wall_config in interior_walls:
            orientation = p.getQuaternionFromEuler([0, 0, wall_config['ori']])
            wall = IndoorWall(
                self.client,
                position=wall_config['pos'],
                orientation=orientation
            )
            self.walls.append(wall)
    
    def _generate_cylinder_obstacles(self):
        """生成圆柱体障碍物随机分布在室内"""
        room_size = self.indoor_config['room_size']
        obstacle_count = self.indoor_config['obstacle_count']
        safe_zone = self.start_config['safe_zone_radius']
        wall_thickness = self.indoor_config['wall_thickness']
        
        # 确保障碍物在房间内，留出安全边距
        safe_margin = 2.0  # 距离墙壁的安全边距
        min_coord = -room_size/2 + safe_margin
        max_coord = room_size/2 - safe_margin
        
        generated_positions = []  # 记录已生成位置，避免重叠
        
        for i in range(obstacle_count):
            max_attempts = 100
            for attempt in range(max_attempts):
                # 在房间内随机生成位置，确保不超出边界
                x = np.random.uniform(min_coord, max_coord)
                y = np.random.uniform(min_coord, max_coord)
                
                # 检查是否在起始安全区域外
                if np.sqrt(x**2 + y**2) <= safe_zone:
                    continue
                    
                # 检查与已有障碍物的距离
                too_close = False
                min_distance = 3.0  # 最小间距3米
                for pos in generated_positions:
                    if np.sqrt((x - pos[0])**2 + (y - pos[1])**2) < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    # 生成圆柱体障碍物
                    obstacle = IndoorCylinder(
                        self.client,
                        position=[x, y, 1.5]  # 高度1.5米
                    )
                    self.obstacles.append(obstacle)
                    generated_positions.append((x, y))
                    break
            else:
                # 如果找不到合适位置，在房间内的网格位置生成
                grid_spacing = room_size / 6  # 网格间距
                row = i // 6
                col = i % 6
                grid_x = (col - 2.5) * grid_spacing
                grid_y = (row - 2.5) * grid_spacing
                
                # 确保在房间内且不在起始区域
                if (abs(grid_x) < room_size/2 - 1 and abs(grid_y) < room_size/2 - 1 and
                    np.sqrt(grid_x**2 + grid_y**2) > safe_zone):
                    obstacle = IndoorCylinder(
                        self.client,
                        position=[grid_x, grid_y, 1.5]
                    )
                    self.obstacles.append(obstacle)
    
    def sample_goal(self, current_positions: List[np.ndarray] = None) -> np.ndarray:
        """采样目标位置"""
        if self.environment_type == "indoor":
            return self._sample_indoor_goal(current_positions)
        else:
            return self._sample_simple_goal(current_positions)
    
    def _sample_indoor_goal(self, current_positions: List[np.ndarray] = None) -> np.ndarray:
        """在室内环境中随机采样目标位置"""
        room_size = self.indoor_config['room_size']
        goal_height = self.goal_config['goal_height']
        safe_margin = self.goal_config['safe_margin']
        safe_zone = self.start_config['safe_zone_radius']
        
        # 随机采样目标位置
        max_attempts = 100
        for attempt in range(max_attempts):
            # 在房间内随机生成位置，保持安全边距
            x = np.random.uniform(-room_size/2 + safe_margin, room_size/2 - safe_margin)
            y = np.random.uniform(-room_size/2 + safe_margin, room_size/2 - safe_margin)
            z = goal_height
            
            # 检查是否在起始安全区域外（避免目标离起点太近）
            distance_from_start = np.sqrt(x**2 + y**2)
            if distance_from_start > safe_zone + 3.0:  # 至少距离起点6米以上
                return np.array([x, y, z])
        
        # 如果找不到合适位置，返回右上角作为默认值
        x = room_size/2 - safe_margin
        y = room_size/2 - safe_margin
        z = goal_height
        
        return np.array([x, y, z])
    
    def _sample_simple_goal(self, current_positions: List[np.ndarray] = None) -> np.ndarray:
        """在简单环境中固定目标位置 - 右上角"""
        goal_height = self.goal_config['goal_height']
        
        # 固定在右上角 (15, 15)
        x = 15.0
        y = 15.0
        z = goal_height
        
        return np.array([x, y, z])
    
    def create_goal_object(self, position: np.ndarray) -> int:
        """创建目标对象"""
        goal = Goal(self.client, position)
        self.goal_id = goal.id
        return goal.id
    
    def set_drone_start_positions(self, drones: List[Any], num_drones: int):
        """设置无人机起始位置"""
        if self.environment_type == "indoor":
            self._set_indoor_start_positions(drones, num_drones)
        else:
            self._set_simple_start_positions(drones, num_drones)
    
    def _set_indoor_start_positions(self, drones: List[Any], num_drones: int):
        """设置室内环境起始位置 - 正方形编队在左下角，朝向目标"""
        formation_spacing = self.start_config['formation_spacing']
        start_height = self.start_config['start_height']
        room_size = self.indoor_config['room_size']
        safe_margin = 2.0  # 安全边距
        
        # 左下角基准位置
        base_x = -room_size/2 + safe_margin
        base_y = -room_size/2 + safe_margin
        
        # 固定朝向：向右（东方，yaw=0）
        target_yaw = 0.0  # 0度表示朝向正东（+X方向）
        orientation = p.getQuaternionFromEuler([0, 0, target_yaw])  # 转换为四元数
        
        # 正方形编队起始位置（相对于基准位置）
        if num_drones == 1:
            positions = [(0.0, 0.0)]
        elif num_drones == 2:
            positions = [(-formation_spacing/2, 0.0), (formation_spacing/2, 0.0)]
        elif num_drones == 3:
            positions = [
                (-formation_spacing, 0.0),
                (formation_spacing, 0.0), 
                (0.0, formation_spacing)
            ]
        elif num_drones == 4:
            positions = [
                (-formation_spacing/2, -formation_spacing/2),
                (formation_spacing/2, -formation_spacing/2),
                (-formation_spacing/2, formation_spacing/2),
                (formation_spacing/2, formation_spacing/2)
            ]
        elif num_drones <= 9:
            # 3x3正方形网格
            positions = []
            side = int(np.ceil(np.sqrt(num_drones)))
            for i in range(num_drones):
                row = i // side
                col = i % side
                x = (col - (side-1)/2) * formation_spacing
                y = (row - (side-1)/2) * formation_spacing
                positions.append((x, y))
        else:
            # 更大的正方形网格
            positions = []
            side = int(np.ceil(np.sqrt(num_drones)))
            for i in range(num_drones):
                row = i // side
                col = i % side
                x = (col - (side-1)/2) * formation_spacing
                y = (row - (side-1)/2) * formation_spacing
                positions.append((x, y))
        
        # 设置无人机位置和朝向（朝向目标）
        for i, (drone, (x, y)) in enumerate(zip(drones, positions)):
            p.resetBasePositionAndOrientation(
                drone.drone,
                [base_x + x, base_y + y, start_height],
                orientation,  # 朝向目标
                physicsClientId=self.client
            )
    
    def _set_simple_start_positions(self, drones: List[Any], num_drones: int):
        """设置简单环境起始位置"""
        formation_spacing = self.start_config['formation_spacing']
        start_height = self.start_config['start_height']
        
        # V型编队起始位置
        for i, drone in enumerate(drones):
            if i == 0:  # 领航者
                x, y = 0.0, 0.0
            else:
                # 跟随者排成V型
                side = 1 if i % 2 == 1 else -1
                row = (i + 1) // 2
                x = side * formation_spacing * row
                y = -formation_spacing * row
            
            p.resetBasePositionAndOrientation(
                drone.drone,
                [x, y, start_height],
                [0, 0, 0, 1],
                physicsClientId=self.client
            )
    
    def _clear_environment(self):
        """清理环境中的对象"""
        # 注意：这里不实际删除PyBullet对象，因为它们在reset时会自动清理
        self.walls.clear()
        self.obstacles.clear()
        self.persons.clear()
        self.goal_id = None


def create_default_environment_config() -> Dict[str, Any]:
    """创建默认环境配置"""
    # 导入配置文件中的设置
    return {
        'environment_type': 'indoor',
        'room_size': 30,  # 30m x 30m的正方形房间
        'wall_height': 3.0,
        'wall_thickness': 0.2,
        'obstacle_count': 16,  # 修复：真正降低到5个圆柱体障碍物，大幅降低难度
        'person_count': 0,  # 删除所有person
        'min_goal_distance': 8.0,  # 适应30m空间
        'max_goal_distance': 15.0,  # 适应30m空间
        'goal_height': config.get('goal_height', 0.85),  # 使用配置文件中的目标高度
        'goal_safe_margin': 2.0,  
        'formation_spacing': 1.0,
        'start_height': config.get('start_height', 0.85),  # 使用配置文件中的起始高度
        'safe_zone_radius': 3.0,  # 适应30m空间的安全区域
    }