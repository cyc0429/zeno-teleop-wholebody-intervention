#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重力补偿力矩计算节点（双机械臂版本）
订阅左右两个机械臂的关节角，计算重力补偿力矩并发布
对于 joint1-3，除以4；对于 joint4-6，保持原样
"""

import rospy
import numpy as np
import pinocchio as pin
import os
import subprocess
from sensor_msgs.msg import JointState


class GravityCompensationArm:
    """单个机械臂的重力补偿节点"""
    
    def __init__(self, arm_side: str = "left"):
        """
        初始化重力补偿节点
        
        Args:
            arm_side: 机械臂侧别，'left' 或 'right'
        """
        if arm_side not in ["left", "right"]:
            raise ValueError("arm_side must be 'left' or 'right'")
        
        self.arm_side = arm_side
        
        # 获取 URDF 文件路径
        try:
            package_path = subprocess.check_output(
                'rospack find piper_description', shell=True
            ).strip().decode('utf-8')
            urdf_path = os.path.join(package_path, 'urdf', 'piper_description.urdf')
            urdf_path = os.path.abspath(urdf_path)
            rospy.loginfo("[%s] URDF path: %s", self.arm_side, urdf_path)
        except Exception as e:
            rospy.logerr("[%s] Failed to find URDF file: %s", self.arm_side, str(e))
            raise
        
        # 加载机器人模型
        try:
            self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
            rospy.loginfo("[%s] Robot model loaded successfully", self.arm_side)
            rospy.loginfo("[%s] Number of joints: %d", self.arm_side, self.robot.model.nq)
        except Exception as e:
            rospy.logerr("[%s] Failed to load robot model: %s", self.arm_side, str(e))
            raise
        
        # 锁定夹爪关节（joint7 和 joint8），只计算前6个关节的重力补偿
        self.joints_to_lock = ["joint7", "joint8"]
        try:
            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.joints_to_lock,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )
            rospy.loginfo("[%s] Reduced robot model created with %d joints", 
                         self.arm_side, self.reduced_robot.model.nq)
        except Exception as e:
            rospy.logerr("[%s] Failed to create reduced robot model: %s", self.arm_side, str(e))
            raise
        
        # 创建数据对象
        self.data = self.reduced_robot.model.createData()
        
        # 订阅关节状态
        self.joint_state_sub = rospy.Subscriber(
            f'/robot/arm_{arm_side}/joint_states_single',
            JointState,
            self.joint_state_callback,
            queue_size=1
        )
        
        # 发布重力补偿力矩
        self.torque_pub = rospy.Publisher(
            f'/robot/arm_{arm_side}/joint_torques_compensated',
            JointState,
            queue_size=1
        )
        
        # 存储关节名称（与输入消息格式一致：6个关节 + gripper）
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        
        rospy.loginfo("[%s] Gravity compensation node initialized", self.arm_side)
        rospy.loginfo("[%s] Subscribing to: /robot/arm_%s/joint_states_single", 
                     self.arm_side, arm_side)
        rospy.loginfo("[%s] Publishing to: /robot/arm_%s/joint_torques_compensated", 
                     self.arm_side, arm_side)
    
    def joint_state_callback(self, msg):
        """关节状态回调函数，计算并发布重力补偿力矩"""
        try:
            # 提取前6个关节的角度（gripper不参与重力补偿计算）
            if len(msg.position) < 6:
                rospy.logwarn("[%s] Received joint state with less than 6 joints", self.arm_side)
                return
            
            # 获取前6个关节的角度
            joint_positions = np.array(msg.position[:6])
            
            # 计算重力补偿力矩（只计算前6个关节）
            gravity_torques = pin.computeGeneralizedGravity(
                self.reduced_robot.model,
                self.data,
                joint_positions
            )
            
            # 应用缩放：joint1-3 除以4，joint4-6 保持原样
            compensated_torques = np.zeros(6)
            compensated_torques[0:3] = gravity_torques[0:3] / 4.0  # joint1-3 除以4
            compensated_torques[3:6] = gravity_torques[3:6]  # joint4-6 保持原样
            
            # 创建输出消息（格式与输入消息一致）
            output_msg = JointState()
            output_msg.header.stamp = rospy.Time.now()
            output_msg.header.frame_id = msg.header.frame_id if msg.header.frame_id else ""
            
            # 设置关节名称（与输入消息格式一致）
            if len(msg.name) >= 7:
                output_msg.name = list(msg.name[:7])
            else:
                output_msg.name = self.joint_names.copy()
            
            # 设置位置（与输入相同）
            if len(msg.position) >= 7:
                output_msg.position = list(msg.position[:7])
            else:
                output_msg.position = list(msg.position[:6]) + [0.0]
            
            # 设置速度（如果有）
            if len(msg.velocity) >= 7:
                output_msg.velocity = list(msg.velocity[:7])
            elif len(msg.velocity) >= 6:
                output_msg.velocity = list(msg.velocity[:6]) + [0.0]
            else:
                output_msg.velocity = [0.0] * 7
            
            # 设置力矩（前6个关节为补偿后的重力补偿力矩，gripper为0）
            output_msg.effort = list(compensated_torques) + [0.0]
            
            # 发布重力补偿力矩消息
            self.torque_pub.publish(output_msg)
            
        except Exception as e:
            rospy.logerr("[%s] Error in joint_state_callback: %s", self.arm_side, str(e))


def check_ros_master():
    """检查 ROS master 是否运行"""
    import rosnode
    try:
        rosnode.rosnode_ping("rosout", max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")


def main():
    """主函数，创建左右两个机械臂的重力补偿节点"""
    try:
        # 检查 ROS master
        check_ros_master()
        
        # 初始化 ROS 节点
        rospy.init_node('piper_gravity_compensation_node', anonymous=True)
        
        # 创建左右两个机械臂的重力补偿节点
        rospy.loginfo("Creating gravity compensation nodes for both arms...")
        arm_left = GravityCompensationArm(arm_side="left")
        arm_right = GravityCompensationArm(arm_side="right")
        
        rospy.loginfo("Gravity compensation nodes for both arms initialized successfully")
        rospy.loginfo("Left arm: joint1-3 divided by 4, joint4-6 unchanged")
        rospy.loginfo("Right arm: joint1-3 divided by 4, joint4-6 unchanged")
        
        # 主循环
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Node failed: %s", str(e))


if __name__ == '__main__':
    main()

