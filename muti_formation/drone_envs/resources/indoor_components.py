import pybullet as p
import os
from .building import Building

def _freeze(body_id: int):
    """将刚体设为静态并清零速度，避免“漂浮/上升”。"""
    try:
        p.changeDynamics(body_id, -1, mass=0)
        p.resetBaseVelocity(body_id, [0,0,0],[0,0,0])
        p.changeDynamics(body_id, -1, linearDamping=1.0, angularDamping=1.0)
    except Exception:
        pass

class IndoorWall(Building):
    """室内墙壁（静态）"""
    def __init__(self, client, position, orientation=[0,0,0,1]):
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indoor_wall.urdf")
        self.id = p.loadURDF(urdf_path, position, orientation, physicsClientId=client)
        _freeze(self.id)
        self.client = client
        self.position = position

class IndoorBox(Building):
    """室内盒子（静态）"""
    def __init__(self, client, position, orientation=[0,0,0,1]):
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indoor_box.urdf")
        self.id = p.loadURDF(urdf_path, position, orientation, physicsClientId=client)
        _freeze(self.id)
        self.client = client
        self.position = position

class IndoorCylinder(Building):
    """室内圆柱体（静态）"""
    def __init__(self, client, position, orientation=[0,0,0,1]):
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indoor_cylinder.urdf")
        self.id = p.loadURDF(urdf_path, position, orientation, physicsClientId=client)
        _freeze(self.id)
        self.client = client
        self.position = position

class IndoorOuterWall(Building):
    """室内外墙（50米长，静态）"""
    def __init__(self, client, position, orientation=[0,0,0,1]):
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indoor_outer_wall.urdf")
        self.id = p.loadURDF(urdf_path, position, orientation, physicsClientId=client)
        _freeze(self.id)
        self.client = client
        self.position = position
