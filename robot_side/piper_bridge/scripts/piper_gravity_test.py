#!/home/zeno/miniconda3/bin python3
# -*- coding: utf-8 -*-
"""
重力补偿力矩计算节点
订阅关节角，计算重力补偿力矩并发布
"""

import rospy
import numpy as np
import pinocchio as pin
import os
import subprocess
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


class GravityCompensationNode:
    def __init__(self):
        """初始化重力补偿节点"""
        rospy.init_node('piper_gravity_compensation_node', anonymous=True)
        
        # 获取 URDF 文件路径
        try:
            package_path = subprocess.check_output(
                'rospack find piper_description', shell=True
            ).strip().decode('utf-8')
            urdf_path = os.path.join(package_path, 'urdf', 'piper_description.urdf')
            urdf_path = os.path.abspath(urdf_path)
            rospy.loginfo("URDF path: %s", urdf_path)
        except Exception as e:
            rospy.logerr("Failed to find URDF file: %s", str(e))
            raise
        
        # 加载机器人模型
        try:
            self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
            rospy.loginfo("Robot model loaded successfully")
            rospy.loginfo("Number of joints: %d", self.robot.model.nq)
        except Exception as e:
            rospy.logerr("Failed to load robot model: %s", str(e))
            raise
        
        # 锁定夹爪关节（joint7 和 joint8），只计算前6个关节的重力补偿
        self.joints_to_lock = ["joint7", "joint8"]
        try:
            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.joints_to_lock,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )
            rospy.loginfo("Reduced robot model created with %d joints", 
                         self.reduced_robot.model.nq)
        except Exception as e:
            rospy.logerr("Failed to create reduced robot model: %s", str(e))
            raise
        
        # 创建数据对象
        self.data = self.reduced_robot.model.createData()
        
        # 订阅关节状态
        self.joint_state_sub = rospy.Subscriber(
            '/teleop/arm_left/joint_states_single',
            JointState,
            self.joint_state_callback,
            queue_size=1
        )
        
        # 发布重力补偿力矩
        self.torque_pub = rospy.Publisher(
            '/teleop/arm_left/joint_torques_estimate',
            JointState,
            queue_size=1
        )
        
        # 发布比例数组
        self.ratio_pub = rospy.Publisher(
            '/teleop/arm_left/effort_ratio',
            Float64MultiArray,
            queue_size=1
        )
        
        # 存储关节名称（与输入消息格式一致：6个关节 + gripper）
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        
        rospy.loginfo("Gravity compensation node initialized")
        rospy.loginfo("Subscribing to: /teleop/arm_left/joint_states_single")
        rospy.loginfo("Publishing to: /teleop/arm_left/joint_torques_estimate")
        rospy.loginfo("Publishing ratio to: /teleop/arm_left/effort_ratio")
    
    def joint_state_callback(self, msg):
        """关节状态回调函数，计算并发布重力补偿力矩"""
        try:
            # 提取前6个关节的角度（gripper不参与重力补偿计算）
            if len(msg.position) < 6:
                rospy.logwarn("Received joint state with less than 6 joints")
                return
            
            # 获取前6个关节的角度
            joint_positions = np.array(msg.position[:6])
            
            # 计算重力补偿力矩（只计算前6个关节）
            gravity_torques = pin.computeGeneralizedGravity(
                self.reduced_robot.model,
                self.data,
                joint_positions
            )
            
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
            
            # 设置力矩（前6个关节为重力补偿力矩，gripper为0）
            output_msg.effort = list(gravity_torques) + [0.0]
            
            # 发布重力补偿力矩消息
            self.torque_pub.publish(output_msg)
            
            # 计算比例：estimated_effort / actual_effort
            # 获取实际的 effort（前6个关节）
            if len(msg.effort) >= 6:
                actual_efforts = np.array(msg.effort[:6])
                estimated_efforts = np.array(gravity_torques)
                
                # 计算比例，避免除零
                ratio_array = np.zeros(6)
                for i in range(6):
                    if abs(actual_efforts[i]) > 1e-6:  # 避免除零
                        ratio_array[i] = estimated_efforts[i] / actual_efforts[i]
                    else:
                        # 如果实际力矩接近零，设置比例为 NaN 或一个特殊值
                        if abs(estimated_efforts[i]) > 1e-6:
                            ratio_array[i] = float('inf')  # 无穷大
                        else:
                            ratio_array[i] = 0.0  # 两者都接近零
                
                # 创建并发布比例数组消息
                ratio_msg = Float64MultiArray()
                ratio_msg.data = list(ratio_array)
                # 设置数组维度信息（可选，但有助于理解数据）
                dim = MultiArrayDimension()
                dim.label = "effort_ratio"
                dim.size = 6
                dim.stride = 6
                ratio_msg.layout.dim.append(dim)
                
                self.ratio_pub.publish(ratio_msg)
            else:
                rospy.logwarn("Received joint state with less than 6 effort values, skipping ratio calculation")
            
        except Exception as e:
            rospy.logerr("Error in joint_state_callback: %s", str(e))


def main():
    try:
        node = GravityCompensationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Node failed: %s", str(e))


if __name__ == '__main__':
    main()

