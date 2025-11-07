import pybullet as p
import os

#这段代码定义了 Goal 类，通过构造函数 __init__ 加载 simplegoal.urdf 文件，在 PyBullet 模拟环境的指定位置创建一个目标对象。
class Goal:
    def __init__(self, client, base):
        #使用 os.path.join 函数将当前脚本所在目录和 simplegoal.urdf 文件名称拼接成完整的文件路径。
        # os.path.dirname(__file__) 获取当前脚本文件所在的目录，simplegoal.urdf 是描述目标对象模型的 URDF（Unified Robot Description Format）文件。
        f_name = os.path.join(os.path.dirname(__file__), 'simplegoal.urdf')
        self.id = p.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], base[2]],
                   physicsClientId=client)


