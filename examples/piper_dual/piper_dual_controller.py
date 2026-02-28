#!/usr/bin/env python3
# -*-coding:utf8-*-
import time
from piper_controller import PiperController


class PiperDualController:
    """
    一个用于控制双臂 Piper 机械臂的集成控制器类。
    """
    def __init__(self, left_can_port="can_left", right_can_port="can_right", gripper_norm=False):
        """
        初始化双臂 Piper 机械臂接口。
        :param left_can_port: 左臂 CAN 端口名称，默认为 "can_left"。
        :param right_can_port: 右臂 CAN 端口名称，默认为 "can_right"。
        :param gripper_norm: 是否对夹爪进行归一化处理。
        """
        self.left_arm = PiperController(left_can_port, gripper_norm)
        self.right_arm = PiperController(right_can_port, gripper_norm)
        print(f"Dual arm controller initialized: Left={left_can_port}, Right={right_can_port}")
    
    def enable(self):
        """
        使能双臂机械臂。
        """
        self.left_arm.enable()
        self.right_arm.enable()
        print("ENABLED: Both robot arms are now enabled.")
    
    def disable(self):
        """
        失能双臂机械臂。
        """
        self.left_arm.disable()
        self.right_arm.disable()
        print("DISABLED: Both robot arms are now disabled.")
    
    def move_end_pose(self, left_position, right_position, velocity=100):
        """
        控制双臂机械臂末端姿态。
        :param left_position: 左臂末端姿态，包含 [X, Y, Z, RX, RY, RZ] 的列表，单位为米和度。
        :param right_position: 右臂末端姿态，包含 [X, Y, Z, RX, RY, RZ] 的列表，单位为米和度。
        :param velocity: 运动速度。
        """
        self.left_arm.move_end_pose(left_position, velocity)
        self.right_arm.move_end_pose(right_position, velocity)
    
    def move_joint(self, left_angles, right_angles, velocity=100):
        """
        控制双臂机械臂关节角度。
        :param left_angles: 左臂关节角度，包含 6 个关节角度的列表，单位为弧度。
        :param right_angles: 右臂关节角度，包含 6 个关节角度的列表，单位为弧度。
        :param velocity: 运动速度。
        """
        self.left_arm.move_joint(left_angles, velocity)
        self.right_arm.move_joint(right_angles, velocity)
    
    def control_gripper(self, left_opening_width, right_opening_width, speed=1000):
        """
        控制双臂夹爪。
        :param left_opening_width: 左臂夹爪开口宽度，单位为米。
        :param right_opening_width: 右臂夹爪开口宽度，单位为米。
        :param speed: 夹爪运动速度。
        """
        self.left_arm.control_gripper(left_opening_width, speed)
        self.right_arm.control_gripper(right_opening_width, speed)
    
    def control_full_joint(self, left_joint, right_joint, velocity=100):
        """
        控制双臂机械臂完整关节（包含夹爪）。
        :param left_joint: 左臂关节角度 + 夹爪，包含 7 个值的列表。
        :param right_joint: 右臂关节角度 + 夹爪，包含 7 个值的列表。
        :param velocity: 运动速度。
        """
        self.left_arm.control_full_joint(left_joint, velocity)
        self.right_arm.control_full_joint(right_joint, velocity)
    
    def control_dual_joint(self, dual_joint, velocity=100):
        """
        使用 14 维向量控制双臂机械臂完整关节（包含夹爪）。
        :param dual_joint: 包含 14 个值的列表，前 7 个为左臂，后 7 个为右臂。
        :param velocity: 运动速度。
        """
        left_joint = dual_joint[0:7]
        right_joint = dual_joint[7:14]
        self.control_full_joint(left_joint, right_joint, velocity)
    
    def get_end_pose(self):
        """
        获取双臂末端姿态信息。
        :return: 左臂和右臂末端姿态信息的元组。
        """
        left_pose = self.left_arm.get_end_pose()
        right_pose = self.right_arm.get_end_pose()
        return left_pose, right_pose
    
    def get_gripper_status(self):
        """
        获取双臂夹爪状态信息。
        :return: 左臂和右臂夹爪状态信息的元组，单位为m。
        """
        left_gripper = self.left_arm.get_gripper_status()
        right_gripper = self.right_arm.get_gripper_status()
        return left_gripper, right_gripper
    
    def get_joint_status(self):
        """
        获取双臂机械臂关节状态信息。
        :return: 左臂和右臂关节状态信息的元组，弧度制。
        """
        left_joints = self.left_arm.get_joint_status()
        right_joints = self.right_arm.get_joint_status()
        return left_joints, right_joints
    
    def get_full_joint_status(self):
        """
        获取双臂机械臂完整关节状态信息。
        :return: 左臂和右臂完整关节状态信息的元组，包含位置和夹爪。
        """
        left_full = self.left_arm.get_full_joint_status()
        right_full = self.right_arm.get_full_joint_status()
        return left_full, right_full
    
    def get_dual_joint_status(self):
        """
        获取双臂机械臂完整关节状态信息（14维向量）。
        :return: 包含 14 个值的列表，前 7 个为左臂，后 7 个为右臂。
        """
        left_full, right_full = self.get_full_joint_status()
        return left_full + right_full


if __name__ == '__main__':
    # 这是一个如何使用 PiperDualController 类的示例
    
    # 1. 初始化双臂控制器
    robot = PiperDualController(left_can_port="can0", right_can_port="can1")

    # 2. 使能双臂机械臂
    robot.enable()

    time.sleep(1)

    # 3. 定义双臂关节目标位置
    left_joints = [
        [0, 0, 0, 0, 0, 0, 0.001],
        [0.2, 0.2, -0.2, 0.3, -0.2, 0.5, 0.08]
    ]
    right_joints = [
        [0, 0, 0, 0, 0, 0, 0.001],
        [-0.2, 0.2, -0.2, -0.3, -0.2, -0.5, 0.08]
    ]

    # 4. 控制双臂运动
    for i in range(10):
        robot.control_full_joint(left_joints[i % 2], right_joints[i % 2])
        left_status, right_status = robot.get_full_joint_status()
        print(f"Left: {left_status}")
        print(f"Right: {right_status}")
        print(f"Dual (14D): {robot.get_dual_joint_status()}")
        time.sleep(2)
    
    # 5. 使用 14 维向量控制
    dual_joint_target = [0, 0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0, 0, 0.05]
    robot.control_dual_joint(dual_joint_target)
    time.sleep(2)

    # 6. 失能双臂机械臂
    robot.disable()

    print("双臂示例程序执行完毕。")
