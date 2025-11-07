import pybullet as p
import os

class Building:
    def __init__(self, client, position, height, urdf_path=None):
        self.client = client
        self.position = position
        self.height = height

        if urdf_path and os.path.exists(urdf_path):
            # 使用URDF文件加载
            self.id = p.loadURDF(
                urdf_path,
                position,
                physicsClientId=client
            )
        else:
            # 回退到直接创建几何体（兼容旧代码）
            width = 4.0
            depth = 4.0
            # 创建视觉形状
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[width/2, depth/2, height/2],
                rgbaColor=[0.5, 0.5, 0.5, 1.0],
                physicsClientId=client
            )
            # 创建碰撞形状
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[width/2, depth/2, height/2],
                physicsClientId=client
            )
            # 创建多体
            self.id = p.createMultiBody(
                baseMass=0,  # 静态物体
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=position,
                physicsClientId=client
            )
