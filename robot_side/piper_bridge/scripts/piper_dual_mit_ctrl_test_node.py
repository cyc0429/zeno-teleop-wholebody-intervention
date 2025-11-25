#!/usr/bin/env python3
# -*-coding:utf8-*-
from typing import (
    Optional,
)
import rospy
import rosnode
from sensor_msgs.msg import JointState
import time
import threading
import math
import numpy as np
from piper_sdk import *
from piper_sdk import C_PiperInterface


def check_ros_master():
    try:
        rosnode.rosnode_ping("rosout", max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")


class C_PiperMitCtrlTestNode:
    """MIT控制测试节点：订阅右臂状态，控制左臂"""

    def __init__(self) -> None:
        check_ros_master()
        # 只在节点未初始化时初始化
        if not rospy.get_node_uri():
            rospy.init_node("piper_dual_mit_ctrl_test_node", anonymous=True)

        # 左臂CAN端口参数
        self.left_can_port = rospy.get_param("~left_can_port", "can0")
        rospy.loginfo("Left CAN port: %s", self.left_can_port)

        # 右臂CAN端口参数
        self.right_can_port = rospy.get_param("~right_can_port", "can1")
        rospy.loginfo("Right CAN port: %s", self.right_can_port)

        # 是否自动使能，默认不自动使能
        self.auto_enable = rospy.get_param("~auto_enable", False)
        rospy.loginfo("Auto enable: %s", self.auto_enable)

        # 左臂MIT控制参数
        self.mit_kp_left = rospy.get_param("~mit_kp_left", 10.0)  # 左臂比例增益，默认10
        self.mit_kd_left = rospy.get_param("~mit_kd_left", 0.8)  # 左臂微分增益，默认0.8
        rospy.loginfo("Left arm MIT control parameters: kp=%.2f, kd=%.2f", self.mit_kp_left, self.mit_kd_left)

        # 右臂MIT控制参数
        self.mit_kp_right = rospy.get_param("~mit_kp_right", 10.0)  # 右臂比例增益，默认10
        self.mit_kd_right = rospy.get_param("~mit_kd_right", 0.8)  # 右臂微分增益，默认0.8
        rospy.loginfo("Right arm MIT control parameters: kp=%.2f, kd=%.2f", self.mit_kp_right, self.mit_kd_right)

        # 力矩缩放系数
        self.torque_scale = rospy.get_param("~torque_scale", 1.0)  # 力矩缩放系数，默认1.0
        rospy.loginfo("Torque scale factor: %.2f", self.torque_scale)

        # 创建左臂piper类并打开can接口
        self.piper_left = C_PiperInterface(can_name=self.left_can_port)
        self.piper_left.ConnectPort()
        # 设置为MIT模式以支持JointMitCtrl
        self.piper_left.MotionCtrl_2(0x01, 0x04, 30, 0xAD)  # MIT mode

        # 创建右臂piper类并打开can接口
        self.piper_right = C_PiperInterface(can_name=self.right_can_port)
        self.piper_right.ConnectPort()
        # 设置为MIT模式以支持JointMitCtrl
        self.piper_right.MotionCtrl_2(0x01, 0x04, 30, 0xAD)  # MIT mode

        # 发布右臂的关节状态
        self.right_joint_pub = rospy.Publisher("/robot/arm_right/joint_states_single", JointState, queue_size=1)

        # 右臂关节状态消息
        self.right_joint_states = JointState()
        self.right_joint_states.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        self.right_joint_states.position = [0.0] * 7
        self.right_joint_states.velocity = [0.0] * 7
        self.right_joint_states.effort = [0.0] * 7

        # 发布左臂的关节状态
        self.left_joint_pub = rospy.Publisher("/robot/arm_left/joint_states_single", JointState, queue_size=1)

        # 左臂关节状态消息
        self.left_joint_states = JointState()
        self.left_joint_states.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        self.left_joint_states.position = [0.0] * 7
        self.left_joint_states.velocity = [0.0] * 7
        self.left_joint_states.effort = [0.0] * 7

        # 订阅robot side右臂的关节状态（自己发布的）
        self.robot_right_joint_state_sub = rospy.Subscriber(
            "/robot/arm_right/joint_states_single",
            JointState,
            self.robot_right_joint_state_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        # 订阅robot side左臂的关节状态（自己发布的）
        self.robot_left_joint_state_sub = rospy.Subscriber(
            "/robot/arm_left/joint_states_single",
            JointState,
            self.robot_left_joint_state_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        # 订阅robot side右臂的重力补偿力矩
        self.robot_right_gravity_torque_sub = rospy.Subscriber(
            "/robot/arm_right/joint_states_compensated",
            JointState,
            self.robot_right_gravity_torque_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        # 订阅robot side左臂的重力补偿力矩
        self.robot_left_gravity_torque_sub = rospy.Subscriber(
            "/robot/arm_left/joint_states_compensated",
            JointState,
            self.robot_left_gravity_torque_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        # 存储数据（线程安全）
        self.robot_joint_state_lock = threading.Lock()
        self.robot_joint_positions_right = [0.0] * 6  # 6个关节的位置
        self.robot_joint_efforts_right = [0.0] * 6  # 6个关节的力矩
        self.robot_joint_positions_left = [0.0] * 6  # 6个关节的位置
        self.robot_joint_efforts_left = [0.0] * 6  # 6个关节的力矩

        # 存储重力补偿力矩（线程安全）
        self.gravity_torques_lock = threading.Lock()
        self.gravity_torques_right = [0.0] * 6  # 右臂6个关节的重力补偿力矩
        self.gravity_torques_left = [0.0] * 6  # 左臂6个关节的重力补偿力矩

        # 控制线程
        self.control_thread_left = None
        self.control_thread_left_running = False
        self.control_thread_right = None
        self.control_thread_right_running = False
        self.__enable_flag_left = False
        self.__enable_flag_right = False

        # 发布线程
        self.publish_thread_right = None
        self.publish_thread_right_running = False
        self.publish_thread_left = None
        self.publish_thread_left_running = False

        rospy.loginfo("Publishing to: /robot/arm_right/joint_states_single")
        rospy.loginfo("Subscribing to: /robot/arm_right/joint_states_single")
        rospy.loginfo("Publishing to: /robot/arm_left/joint_states_single")
        rospy.loginfo("Subscribing to: /robot/arm_left/joint_states_single")
        rospy.loginfo("Subscribing to: /robot/arm_right/joint_torques_compensated")
        rospy.loginfo("Subscribing to: /robot/arm_left/joint_torques_compensated")
        rospy.loginfo("Right arm: MIT control using left arm joint angles")
        rospy.loginfo("Left arm: MIT control using right arm effort minus left gravity compensation")

    def robot_right_joint_state_callback(self, msg):
        """Robot side右臂关节状态回调函数"""
        if len(msg.position) >= 6 and len(msg.effort) >= 6:
            with self.robot_joint_state_lock:
                self.robot_joint_positions_right = list(msg.position[:6])
                self.robot_joint_efforts_right = list(msg.effort[:6])

    def robot_left_joint_state_callback(self, msg):
        """Robot side左臂关节状态回调函数"""
        if len(msg.position) >= 6 and len(msg.effort) >= 6:
            with self.robot_joint_state_lock:
                self.robot_joint_positions_left = list(msg.position[:6])
                self.robot_joint_efforts_left = list(msg.effort[:6])

    def robot_right_gravity_torque_callback(self, msg):
        """Robot side右臂重力补偿力矩回调函数"""
        if len(msg.effort) >= 6:
            with self.gravity_torques_lock:
                self.gravity_torques_right = list(msg.effort[:6])

    def robot_left_gravity_torque_callback(self, msg):
        """Robot side左臂重力补偿力矩回调函数"""
        if len(msg.effort) >= 6:
            with self.gravity_torques_lock:
                self.gravity_torques_left = list(msg.effort[:6])

    def ControlThreadRight(self):
        """右臂MIT控制线程，使用左臂的关节角作为位置参考"""
        rate = rospy.Rate(200)  # 200 Hz

        while not rospy.is_shutdown() and self.control_thread_right_running:
            if not self.__enable_flag_right:
                rate.sleep()
                continue

            # 获取左臂的关节角数据
            with self.robot_joint_state_lock:
                left_positions = self.robot_joint_positions_left[:]

            # 限制位置范围到[-12.5, 12.5] rad
            for i in range(6):
                left_positions[i] = max(-12.5, min(12.5, left_positions[i]))

            # 使用JointMitCtrl控制右臂每个关节
            try:
                for motor_num in range(1, 7):
                    joint_idx = motor_num - 1
                    self.piper_right.JointMitCtrl(
                        motor_num=motor_num,
                        pos_ref=left_positions[joint_idx],
                        vel_ref=0.0,  # 速度参考值设置为0
                        kp=self.mit_kp_right,
                        kd=self.mit_kd_right,
                        t_ref=0.0,  # 前馈力矩设置为0
                    )
            except Exception as e:
                rospy.logerr("Error in JointMitCtrl (right arm): %s", str(e))

            rate.sleep()

    def ControlThreadLeft(self):
        """左臂MIT控制线程，使用右臂的effort减去左臂重力补偿后乘以缩放系数"""
        rate = rospy.Rate(200)  # 200 Hz

        while not rospy.is_shutdown() and self.control_thread_left_running:
            if not self.__enable_flag_left:
                rate.sleep()
                continue

            # 获取右臂的effort和左臂的重力补偿力矩
            with self.robot_joint_state_lock:
                right_efforts = self.robot_joint_efforts_right[:]
                left_positions = self.robot_joint_positions_left[:]

            with self.gravity_torques_lock:
                left_gravity_torques = self.gravity_torques_left[:]

            # 计算目标力矩：右臂effort - 左臂重力补偿，然后乘以缩放系数
            target_torques = [0.0] * 6
            for i in range(6):
                # 右臂effort减去左臂重力补偿
                interaction_torque = right_efforts[i] - left_gravity_torques[i]
                # 乘以缩放系数
                target_torques[i] = -interaction_torque * self.torque_scale + left_gravity_torques[i]
                # 限制力矩范围到[-18.0, 18.0] N·m
                target_torques[i] = max(-18.0, min(18.0, target_torques[i]))

            # 限制位置范围到[-12.5, 12.5] rad
            for i in range(6):
                left_positions[i] = max(-12.5, min(12.5, left_positions[i]))

            # 使用JointMitCtrl控制左臂每个关节
            try:
                for motor_num in range(1, 7):
                    joint_idx = motor_num - 1
                    self.piper_left.JointMitCtrl(
                        motor_num=motor_num,
                        pos_ref=left_positions[joint_idx],
                        vel_ref=0.0,  # 速度参考值设置为0
                        kp=self.mit_kp_left,
                        kd=self.mit_kd_left,
                        t_ref=target_torques[joint_idx],
                    )
            except Exception as e:
                rospy.logerr("Error in JointMitCtrl (left arm): %s", str(e))

            rate.sleep()

    def PublishRightArmJointStates(self):
        """发布右臂关节状态线程"""
        rate = rospy.Rate(200)  # 200 Hz

        while not rospy.is_shutdown() and self.publish_thread_right_running:
            # 获取右臂关节角和夹爪位置
            # 由于获取的原始数据是度为单位扩大了1000倍，因此要转为弧度需要先除以1000，再乘3.14/180
            joint_0: float = (self.piper_right.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
            joint_1: float = (self.piper_right.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
            joint_2: float = (self.piper_right.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
            joint_3: float = (self.piper_right.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
            joint_4: float = (self.piper_right.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
            joint_5: float = (self.piper_right.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
            joint_6: float = self.piper_right.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
            vel_0: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000
            vel_1: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000
            vel_2: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000
            vel_3: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000
            vel_4: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000
            vel_5: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000
            effort_0: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_1.effort / 1000
            effort_1: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_2.effort / 1000
            effort_2: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_3.effort / 1000
            effort_3: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_4.effort / 1000
            effort_4: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_5.effort / 1000
            effort_5: float = self.piper_right.GetArmHighSpdInfoMsgs().motor_6.effort / 1000
            effort_6: float = self.piper_right.GetArmGripperMsgs().gripper_state.grippers_effort / 1000

            self.right_joint_states.header.stamp = rospy.Time.now()
            self.right_joint_states.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
            self.right_joint_states.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0]
            self.right_joint_states.effort = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5, effort_6]

            # 发布右臂关节状态
            self.right_joint_pub.publish(self.right_joint_states)

            rate.sleep()

    def PublishLeftArmJointStates(self):
        """发布左臂关节状态线程"""
        rate = rospy.Rate(200)  # 200 Hz

        while not rospy.is_shutdown() and self.publish_thread_left_running:
            # 获取左臂关节角和夹爪位置
            # 由于获取的原始数据是度为单位扩大了1000倍，因此要转为弧度需要先除以1000，再乘3.14/180
            joint_0: float = (self.piper_left.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
            joint_1: float = (self.piper_left.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
            joint_2: float = (self.piper_left.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
            joint_3: float = (self.piper_left.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
            joint_4: float = (self.piper_left.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
            joint_5: float = (self.piper_left.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
            joint_6: float = self.piper_left.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
            vel_0: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000
            vel_1: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000
            vel_2: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000
            vel_3: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000
            vel_4: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000
            vel_5: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000
            effort_0: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_1.effort / 1000
            effort_1: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_2.effort / 1000
            effort_2: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_3.effort / 1000
            effort_3: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_4.effort / 1000
            effort_4: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_5.effort / 1000
            effort_5: float = self.piper_left.GetArmHighSpdInfoMsgs().motor_6.effort / 1000
            effort_6: float = self.piper_left.GetArmGripperMsgs().gripper_state.grippers_effort / 1000

            self.left_joint_states.header.stamp = rospy.Time.now()
            self.left_joint_states.position = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
            self.left_joint_states.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0]
            self.left_joint_states.effort = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5, effort_6]

            # 发布左臂关节状态
            self.left_joint_pub.publish(self.left_joint_states)

            rate.sleep()

    def EnableLeftArm(self):
        """使能左臂"""
        enable_flag = False
        timeout = 5
        start_time = time.time()

        while not enable_flag:
            elapsed_time = time.time() - start_time
            rospy.loginfo("--------------------[Left Arm]--------------------")
            enable_flag = (
                self.piper_left.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
                and self.piper_left.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
                and self.piper_left.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
                and self.piper_left.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
                and self.piper_left.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
                and self.piper_left.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            )
            rospy.loginfo("左臂使能状态: %s", enable_flag)
            self.piper_left.EnableArm(7)
            self.piper_left.GripperCtrl(0, 1000, 0x01, 0)
            if enable_flag:
                self.__enable_flag_left = True
            rospy.loginfo("--------------------[Left Arm]--------------------")

            if elapsed_time > timeout:
                rospy.logwarn("左臂使能超时")
                break
            time.sleep(1)

    def EnableRightArm(self):
        """使能右臂"""
        enable_flag = False
        timeout = 5
        start_time = time.time()

        while not enable_flag:
            elapsed_time = time.time() - start_time
            rospy.loginfo("--------------------[Right Arm]--------------------")
            enable_flag = (
                self.piper_right.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
                and self.piper_right.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
                and self.piper_right.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
                and self.piper_right.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
                and self.piper_right.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
                and self.piper_right.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            )
            rospy.loginfo("右臂使能状态: %s", enable_flag)
            self.piper_right.EnableArm(7)
            self.piper_right.GripperCtrl(0, 1000, 0x01, 0)
            if enable_flag:
                self.__enable_flag_right = True
            rospy.loginfo("--------------------[Right Arm]--------------------")

            if elapsed_time > timeout:
                rospy.logwarn("右臂使能超时")
                break
            time.sleep(1)

    def Run(self):
        """运行节点"""
        # 如果设置了自动使能，则使能左右两个机械臂
        if self.auto_enable:
            # 在单独的线程中使能左右臂
            left_enable_thread = threading.Thread(target=self.EnableLeftArm, daemon=True)
            right_enable_thread = threading.Thread(target=self.EnableRightArm, daemon=True)
            left_enable_thread.start()
            right_enable_thread.start()

        # 启动发布右臂关节状态的线程
        self.publish_thread_right_running = True
        self.publish_thread_right = threading.Thread(target=self.PublishRightArmJointStates, daemon=True)
        self.publish_thread_right.start()
        rospy.loginfo("Right arm joint states publish thread started")

        # 启动发布左臂关节状态的线程
        self.publish_thread_left_running = True
        self.publish_thread_left = threading.Thread(target=self.PublishLeftArmJointStates, daemon=True)
        self.publish_thread_left.start()
        rospy.loginfo("Left arm joint states publish thread started")

        # 启动右臂MIT控制线程
        self.control_thread_right_running = True
        self.control_thread_right = threading.Thread(target=self.ControlThreadRight, daemon=True)
        self.control_thread_right.start()
        rospy.loginfo("Right arm MIT control thread started")

        # 启动左臂MIT控制线程
        self.control_thread_left_running = True
        self.control_thread_left = threading.Thread(target=self.ControlThreadLeft, daemon=True)
        self.control_thread_left.start()
        rospy.loginfo("Left arm MIT control thread started")

        # 主线程保持运行
        rospy.spin()


if __name__ == "__main__":
    try:
        check_ros_master()
        node = C_PiperMitCtrlTestNode()
        node.Run()
    except rospy.ROSInterruptException:
        pass
