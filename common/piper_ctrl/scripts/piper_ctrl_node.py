#!/usr/bin/env python3
# -*-coding:utf8-*-
from typing import Optional
import rospy
import rosnode
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse
import time
import threading
import math
import numpy as np
from piper_sdk import *
from piper_sdk import C_PiperInterface
from std_srvs.srv import Trigger, TriggerResponse
from piper_msgs.msg import PiperStatusMsg, PiperEulerPose
from piper_msgs.srv import Enable, EnableResponse
from piper_msgs.srv import Gripper, GripperResponse
from piper_msgs.srv import GoZero, GoZeroResponse
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler


def check_ros_master():
    try:
        rosnode.rosnode_ping("rosout", max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")


class C_PiperCtrlNode:
    """Unified robot arm control node supporting position and MIT control modes"""

    def __init__(self) -> None:
        check_ros_master()
        if not rospy.get_node_uri():
            rospy.init_node("piper_ctrl_node", anonymous=True)

        # Basic parameters
        self.can_port = rospy.get_param("~can_port", "can0")
        rospy.loginfo("CAN port: %s", self.can_port)

        self.topic_prefix = rospy.get_param("~topic_prefix", "/")
        if not self.topic_prefix.endswith("/"):
            self.topic_prefix += "/"
        rospy.loginfo("Topic prefix: %s", self.topic_prefix)

        self.auto_enable = rospy.get_param("~auto_enable", False)
        rospy.loginfo("Auto enable: %s", self.auto_enable)

        self.gripper_exist = rospy.get_param("~gripper_exist", True)
        rospy.loginfo("Gripper exist: %s", self.gripper_exist)

        self.gripper_val_mutiple = rospy.get_param("~gripper_val_mutiple", 1.0)
        if self.gripper_val_mutiple <= 0:
            rospy.logwarn("Invalid gripper_val_mutiple value: must be positive. Using default value of 1.")
            self.gripper_val_mutiple = 1.0
        rospy.loginfo("Gripper value multiple: %.2f", self.gripper_val_mutiple)

        self.enable_gripper = rospy.get_param("~enable_gripper", True)
        rospy.loginfo("Enable gripper: %s", self.enable_gripper)

        self.enable_gripper_haptic = rospy.get_param("~enable_gripper_haptic", False)
        rospy.loginfo("Enable gripper haptic: %s", self.enable_gripper_haptic)

        self.gripper_range = rospy.get_param("~gripper_range", 0.1)
        rospy.loginfo("Gripper range: %.2f", self.gripper_range)

        self.gripper_reverse = rospy.get_param("~gripper_reverse", False)
        rospy.loginfo("Gripper reverse: %s", self.gripper_reverse)

        # Control mode parameters
        self.ctrl_mode = rospy.get_param("~ctrl_mode", "p").lower()
        if self.ctrl_mode not in ["p", "mit"]:
            rospy.logwarn("Invalid ctrl_mode: %s. Using default 'p'", self.ctrl_mode)
            self.ctrl_mode = "p"
        rospy.loginfo("Control mode: %s", self.ctrl_mode)

        # Position control parameters
        self.p_speed = rospy.get_param("~p/speed", 50)
        self.p_speed = max(0, min(100, self.p_speed))
        rospy.loginfo("Position control speed: %d", self.p_speed)

        # MIT control parameters
        self.mit_speed = rospy.get_param("~mit/speed", 50)
        self.mit_speed = max(0, min(100, self.mit_speed))

        # kp parameter
        kp_param = rospy.get_param("~mit/kp", 10.0)
        if isinstance(kp_param, list):
            if len(kp_param) == 6:
                self.mit_kp = list(kp_param)
            else:
                rospy.logwarn("kp list length should be 6. Using first value for all joints.")
                self.mit_kp = [kp_param[0] if len(kp_param) > 0 else 10.0] * 6
        else:
            self.mit_kp = [float(kp_param)] * 6

        # kd parameter
        kd_param = rospy.get_param("~mit/kd", 0.8)
        if isinstance(kd_param, list):
            if len(kd_param) == 6:
                self.mit_kd = list(kd_param)
            else:
                rospy.logwarn("kd list length should be 6. Using first value for all joints.")
                self.mit_kd = [kd_param[0] if len(kd_param) > 0 else 0.8] * 6
        else:
            self.mit_kd = [float(kd_param)] * 6

        self.mit_enable_pos = rospy.get_param("~mit/enable_pos", True)
        self.mit_enable_vel = rospy.get_param("~mit/enable_vel", True)
        self.mit_enable_tor = rospy.get_param("~mit/enable_tor", True)
        self.mit_enable_gravity = rospy.get_param("~mit/enable_gravity", False)
        torque_scale_param = rospy.get_param("~mit/torque_scale", 1.0)
        if isinstance(torque_scale_param, list):
            if len(torque_scale_param) == 6:
                self.mit_torque_scale = list(torque_scale_param)
            else:
                rospy.logwarn("torque_scale list length should be 6. Using first value for all joints.")
                self.mit_torque_scale = [torque_scale_param[0] if len(torque_scale_param) > 0 else 1.0] * 6
        else:
            self.mit_torque_scale = [float(torque_scale_param)] * 6

        # Topic name parameters
        def ensure_topic_prefix(param_name, default_suffix):
            if rospy.has_param(param_name):
                topic = rospy.get_param(param_name)
                topic = topic.lstrip("/")
                return self.topic_prefix + topic
            else:
                return self.topic_prefix + default_suffix

        def get_remap_topic(remap_param_name, default_topic):
            if rospy.has_param(remap_param_name):
                return rospy.get_param(remap_param_name)
            return default_topic

        # Build default topic names
        default_p_joint_pos_cmd_topic = ensure_topic_prefix("~p/joint_pos_cmd_topic", "joint_pos_cmd")
        default_mit_joint_pos_cmd_topic = ensure_topic_prefix("~mit/joint_pos_cmd_topic", "joint_pos_cmd")
        default_mit_joint_vel_cmd_topic = ensure_topic_prefix("~mit/joint_vel_cmd_topic", "joint_vel_cmd")
        default_mit_joint_tor_cmd_topic = ensure_topic_prefix("~mit/joint_tor_cmd_topic", "joint_tor_cmd")
        default_gripper_pos_cmd_topic = ensure_topic_prefix("~gripper_pos_cmd_topic", "gripper_pos_cmd")
        default_gripper_effort_cmd_topic = ensure_topic_prefix("~gripper_effort_cmd_topic", "gripper_effort_cmd")

        # Apply remap if configured
        self.p_joint_pos_cmd_topic = get_remap_topic("~remap/joint_pos_cmd_to", default_p_joint_pos_cmd_topic)
        self.mit_joint_pos_cmd_topic = get_remap_topic("~remap/joint_pos_cmd_to", default_mit_joint_pos_cmd_topic)
        self.mit_joint_vel_cmd_topic = get_remap_topic("~remap/joint_vel_cmd_to", default_mit_joint_vel_cmd_topic)
        self.mit_joint_tor_cmd_topic = get_remap_topic("~remap/joint_tor_cmd_to", default_mit_joint_tor_cmd_topic)
        self.gripper_pos_cmd_topic = get_remap_topic("~remap/gripper_pos_cmd_to", default_gripper_pos_cmd_topic)
        self.gripper_effort_cmd_topic = get_remap_topic("~remap/gripper_effort_cmd_to", default_gripper_effort_cmd_topic)

        if self.mit_enable_gravity:
            default_joint_states_compensated_topic = ensure_topic_prefix(
                "~mit/joint_states_compensated_topic", "joint_states_compensated"
            )
            self.mit_joint_states_compensated_topic = get_remap_topic(
                "~remap/joint_states_compensated_to", default_joint_states_compensated_topic
            )

        # Thread rate parameters
        self.publish_rate = rospy.get_param("~publish_rate", 200.0)
        self.control_rate = rospy.get_param("~control_rate", 200.0)
        self.subscribe_rate = rospy.get_param("~subscribe_rate", 100.0)

        # Filter parameters
        self.filter_enable = rospy.get_param("~filter/enable", True)
        self.filter_alpha_pos = rospy.get_param("~filter/alpha_position", 0.7)
        self.filter_alpha_vel = rospy.get_param("~filter/alpha_velocity", 0.5)
        self.filter_alpha_effort = rospy.get_param("~filter/alpha_effort", 0.5)
        self.filter_alpha_pos = max(0.0, min(1.0, self.filter_alpha_pos))
        self.filter_alpha_vel = max(0.0, min(1.0, self.filter_alpha_vel))
        self.filter_alpha_effort = max(0.0, min(1.0, self.filter_alpha_effort))

        # Create piper interface
        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()

        if self.ctrl_mode == "mit":
            self.piper.MotionCtrl_2(0x01, 0x04, self.mit_speed, 0xAD)
        else:
            self.piper.MotionCtrl_2(0x01, 0x01, self.p_speed, 0)

        self.block_ctrl_flag = False
        self.__enable_flag = False

        # Publishers
        self.joint_pub = rospy.Publisher(self.topic_prefix + "joint_states_single", JointState, queue_size=1)
        self.arm_status_pub = rospy.Publisher(self.topic_prefix + "arm_status", PiperStatusMsg, queue_size=1)
        self.end_pose_pub = rospy.Publisher(self.topic_prefix + "end_pose", PoseStamped, queue_size=1)
        self.end_pose_euler_pub = rospy.Publisher(self.topic_prefix + "end_pose_euler", PiperEulerPose, queue_size=1)

        # Services
        self.enable_service = rospy.Service(self.topic_prefix + "enable_srv", Enable, self.handle_enable_service)
        self.gripper_service = rospy.Service(self.topic_prefix + "gripper_srv", Gripper, self.handle_gripper_service)
        self.stop_service = rospy.Service(self.topic_prefix + "stop_srv", Trigger, self.handle_stop_service)
        self.reset_service = rospy.Service(self.topic_prefix + "reset_srv", Trigger, self.handle_reset_service)
        self.go_zero_service = rospy.Service(self.topic_prefix + "go_zero_srv", GoZero, self.handle_go_zero_service)
        self.block_arm_service = rospy.Service(self.topic_prefix + "block_arm", SetBool, self.handle_block_arm_service)

        # Joint state message
        self.joint_states = JointState()
        self.joint_states.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        self.joint_states.position = [0.0] * 7
        self.joint_states.velocity = [0.0] * 7
        self.joint_states.effort = [0.0] * 7

        # Control data storage (thread-safe)
        self.control_data_lock = threading.Lock()
        self.joint_positions_cmd = [0.0] * 7
        self.joint_velocities_cmd = [0.0] * 6
        self.joint_torques_cmd = [0.0] * 6
        self.current_joint_positions = [0.0] * 6
        self.current_joint_velocities = [0.0] * 6
        self.current_joint_efforts = [0.0] * 6

        self.gravity_torques_lock = threading.Lock()
        self.gravity_torques = [0.0] * 6

        self.gripper_cmd_lock = threading.Lock()
        self.gripper_cmd_position = 0.0
        self.gripper_cmd_effort = 0.0

        self.filter_lock = threading.Lock()
        self.filtered_positions = [0.0] * 7
        self.filtered_velocities = [0.0] * 7
        self.filtered_efforts = [0.0] * 7
        self.filter_initialized = False

        # =========================================================================
        # ✅ 主从遥操、接管锁与“虚拟悬停”状态变量
        # =========================================================================
        self.master_slave_enable = rospy.get_param("~master_slave/enable", False)
        self.slave_follow_mode = False
        self.master_joint_positions = [0.0] * 6
        self.master_gripper_position = 0.0  
        self.master_gripper_effort = 0.0    
        
        self.takeover_locked = False
        self.locked_gripper_pos = 0.0
        self.current_gripper_pos = 0.0
        self.prev_robot_mode_is_inference = False
        
        kp_follow_param = rospy.get_param("~master_slave/kp_follow", [10.0] * 6)
        if isinstance(kp_follow_param, list) and len(kp_follow_param) == 6:
            self.kp_follow = list(kp_follow_param)
        else:
            self.kp_follow = [float(kp_follow_param[0] if isinstance(kp_follow_param, list) and len(kp_follow_param) > 0 else 10.0)] * 6

        takeover_topic = rospy.get_param("~master_slave/master_flag_topic", "/teleop/slave_follow_flag")
        rospy.Subscriber(takeover_topic, Bool, self._takeover_flag_cb, queue_size=1, tcp_nodelay=True)

        if self.master_slave_enable:
            master_pos_topic = rospy.get_param("~master_slave/master_position_topic", "/robot/arm_left/joint_states_single")
            rospy.Subscriber(master_pos_topic, JointState, self._master_pos_cb, queue_size=1, tcp_nodelay=True)
        # =========================================================================

        # Subscribers
        if self.ctrl_mode == "p":
            rospy.Subscriber(
                self.p_joint_pos_cmd_topic, JointState, self.joint_pos_cmd_callback, queue_size=1, tcp_nodelay=True
            )
        else:
            if self.mit_enable_pos:
                rospy.Subscriber(
                    self.mit_joint_pos_cmd_topic, JointState, self.joint_pos_cmd_callback, queue_size=1, tcp_nodelay=True
                )
            if self.mit_enable_vel:
                rospy.Subscriber(
                    self.mit_joint_vel_cmd_topic, JointState, self.joint_vel_cmd_callback, queue_size=1, tcp_nodelay=True
                )
            if self.mit_enable_tor:
                rospy.Subscriber(
                    self.mit_joint_tor_cmd_topic, JointState, self.joint_tor_cmd_callback, queue_size=1, tcp_nodelay=True
                )
            if self.mit_enable_gravity:
                rospy.Subscriber(
                    self.mit_joint_states_compensated_topic, JointState, self.gravity_torque_callback, queue_size=1, tcp_nodelay=True
                )

        rospy.Subscriber(self.topic_prefix + "enable_flag", Bool, self.enable_callback, queue_size=1, tcp_nodelay=True)

        # Gripper command subscriber
        if self.enable_gripper:
            rospy.Subscriber(
                self.gripper_pos_cmd_topic, JointState, self.gripper_pos_cmd_callback, queue_size=1, tcp_nodelay=True
            )
            rospy.Subscriber(
                self.gripper_effort_cmd_topic, JointState, self.gripper_effort_cmd_callback, queue_size=1, tcp_nodelay=True
            )

        # Thread control flags
        self.publish_thread_running = False
        self.control_thread_running = False
        self.publish_thread = None
        self.control_thread = None

    def GetEnableFlag(self):
        return self.__enable_flag

    def _takeover_flag_cb(self, msg: Bool):
        is_inference = msg.data
        self.slave_follow_mode = is_inference
        
        # 从推理切回遥操时上锁 (防掉落)
        if not is_inference and self.prev_robot_mode_is_inference:
            if not self.master_slave_enable: 
                self.takeover_locked = True
                self.locked_gripper_pos = self.current_gripper_pos
                rospy.loginfo("[%s] ⚠️ 夹爪已上锁 (%.3f)！请捏合遥控夹爪至相同位置以解锁接管！", self.topic_prefix, self.locked_gripper_pos)

        self.prev_robot_mode_is_inference = is_inference

    def _master_pos_cb(self, msg: JointState):
        if len(msg.position) >= 6:
            with self.control_data_lock:
                self.master_joint_positions = list(msg.position[:6])
                if len(msg.position) >= 7:
                    self.master_gripper_position = msg.position[6]
                if len(msg.effort) >= 7:
                    self.master_gripper_effort = msg.effort[6]

    # Callback functions
    def joint_pos_cmd_callback(self, msg: JointState):
        if not self.block_ctrl_flag and len(msg.position) >= 6:
            with self.control_data_lock:
                self.joint_positions_cmd[:6] = list(msg.position[:6])
                if len(msg.position) >= 7:
                    self.joint_positions_cmd[6] = msg.position[6]

    def joint_vel_cmd_callback(self, msg: JointState):
        if not self.block_ctrl_flag and len(msg.velocity) >= 6:
            with self.control_data_lock:
                self.joint_velocities_cmd = list(msg.velocity[:6])

    def joint_tor_cmd_callback(self, msg: JointState):
        if not self.block_ctrl_flag and len(msg.effort) >= 6:
            with self.control_data_lock:
                self.joint_torques_cmd = list(msg.effort[:6])

    def gravity_torque_callback(self, msg: JointState):
        if len(msg.effort) >= 6:
            with self.gravity_torques_lock:
                self.gravity_torques = list(msg.effort[:6])

    def gripper_pos_cmd_callback(self, msg: JointState):
        if not self.block_ctrl_flag:
            with self.gripper_cmd_lock:
                if len(msg.position) >= 7:
                    self.gripper_cmd_position = msg.position[6]

    def gripper_effort_cmd_callback(self, msg: JointState):
        if not self.block_ctrl_flag:
            with self.gripper_cmd_lock:
                if len(msg.effort) >= 7:
                    self.gripper_cmd_effort = msg.effort[6]

    def enable_callback(self, enable_flag: Bool):
        if enable_flag.data:
            self.__enable_flag = True
            self.piper.EnableArm(7)
            if self.gripper_exist:
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self.__enable_flag = False
            self.piper.DisableArm(7)
            if self.gripper_exist:
                self.piper.GripperCtrl(0, 1000, 0x00, 0)

    # Control thread
    def ControlThread(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown() and self.control_thread_running:
            if not self.__enable_flag:
                rate.sleep()
                continue
            try:
                if self.ctrl_mode == "p":
                    self._position_control()
                else:
                    self._mit_control()
            except Exception as e:
                rospy.logerr("Error in control thread: %s", str(e))
            rate.sleep()

    def _position_control(self):
        with self.control_data_lock:
            positions = self.joint_positions_cmd[:]
            has_gripper = len(self.joint_positions_cmd) >= 7
            gripper_pos = self.joint_positions_cmd[6] if has_gripper else 0.0

        for i in range(6):
            positions[i] = max(-12.5, min(12.5, positions[i]))

        factor = 1000 * 180 / np.pi
        joint_0 = round(positions[0] * factor)
        joint_1 = round(positions[1] * factor)
        joint_2 = round(positions[2] * factor)
        joint_3 = round(positions[3] * factor)
        joint_4 = round(positions[4] * factor)
        joint_5 = round(positions[5] * factor)

        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

        if self.enable_gripper and self.gripper_exist:
            if has_gripper:
                raw_cmd_pos = gripper_pos
            else:
                with self.gripper_cmd_lock:
                    raw_cmd_pos = self.gripper_cmd_position

            gripper_target = raw_cmd_pos

            # 接管锁逻辑
            if self.takeover_locked:
                if abs(gripper_target - self.locked_gripper_pos) < 0.005:
                    self.takeover_locked = False
                    rospy.loginfo("[%s] ✅ 夹爪已对齐，成功解除接管锁定！", self.topic_prefix)
                else:
                    gripper_target = self.locked_gripper_pos

            # =========================================================================
            # ✅ “悬停+放大力反馈”模式：松手保持原位，夹物增强感受
            # =========================================================================
            if self.master_slave_enable:
                if self.slave_follow_mode:
                    # 推理模式：锁定自己跟随 Robot
                    with self.control_data_lock:
                        target_pos = self.master_gripper_position
                    if self.gripper_reverse:
                        target_pos = self.gripper_range - target_pos
                    joint_6 = round(target_pos * 1000 * 1000 * self.gripper_val_mutiple)
                    gripper_effort = round(3.0 * 1000)
                else:
                    # 遥操模式：
                    with self.control_data_lock:
                        robot_pos = self.master_gripper_position
                        robot_eff = abs(self.master_gripper_effort)

                    # --- 参数调优区 ---
                    force_gain = 2.5       # 力矩放大系数：把Robot的微小阻力放大给你的手 (你可以调到3或4)
                    eff_threshold = 0.25   # 阈值：当阻力大于0.25N时才认为“夹到东西了”
                    # -----------------

                    if robot_eff > eff_threshold:
                        # 状态A【夹到硬物】：强制将目标设为Robot当前位置，把你的手顶住
                        target_pos = robot_pos
                        if self.gripper_reverse:
                            target_pos = self.gripper_range - target_pos
                        joint_6 = round(target_pos * 1000 * 1000 * self.gripper_val_mutiple)
                        
                        # 成倍放大反馈力量，手感更猛
                        feedback_effort = robot_eff * force_gain
                        feedback_effort = max(0.5, min(feedback_effort, 3.0))
                        gripper_effort = round(feedback_effort * 1000)
                    else:
                        # 状态B【空气中空抓 / 松开手】：目标设为遥操端【自己的当前物理位置】
                        # 从而完美实现松手悬停，不再自动退回0度！
                        target_pos = self.current_gripper_pos
                        if self.gripper_reverse:
                            target_pos = self.gripper_range - target_pos
                        joint_6 = round(target_pos * 1000 * 1000 * self.gripper_val_mutiple)
                        
                        # 0.15N的微弱电流作为阻尼，足以抵抗重力悬停，又方便人手捏合滑动
                        gripper_effort = round(0.15 * 1000)
            else:
                # 机器端保持原生控制逻辑
                joint_6 = round(gripper_target * 1000 * 1000 * self.gripper_val_mutiple)
                if self.enable_gripper_haptic:
                    with self.gripper_cmd_lock:
                        gripper_effort = abs(self.gripper_cmd_effort)
                        gripper_effort = max(0, min(gripper_effort, 3))
                        gripper_effort = round(gripper_effort * 1000)
                else:
                    gripper_effort = round(1000 * 1.0)
            # =========================================================================

            joint_6 = max(0, min(80000, joint_6))
            if abs(joint_6) < 200:
                joint_6 = 0

            if gripper_effort < 50:
                self.piper.GripperCtrl(abs(joint_6), gripper_effort, 0x00, 0)
            else:
                self.piper.GripperCtrl(abs(joint_6), gripper_effort, 0x01, 0)

    def _mit_control(self):
        with self.control_data_lock:
            if self.mit_enable_pos:
                positions = self.joint_positions_cmd[:]
            else:
                positions = self.current_joint_positions[:]

            if self.mit_enable_vel:
                velocities = self.joint_velocities_cmd[:]
            else:
                velocities = [0.0] * 6

            if self.mit_enable_tor:
                torques = self.joint_torques_cmd[:]
            else:
                torques = [0.0] * 6

        if self.mit_enable_gravity:
            with self.gravity_torques_lock:
                gravity_torques = self.gravity_torques[:]
            for i in range(6):
                torques[i] = -(torques[i] - gravity_torques[i]) * self.mit_torque_scale[i] + gravity_torques[i]
        else:
            for i in range(6):
                torques[i] = torques[i] * self.mit_torque_scale[i]

        for i in range(6):
            positions[i] = max(-12.5, min(12.5, positions[i]))
            velocities[i] = max(-45.0, min(45.0, velocities[i]))
            torques[i] = max(-18.0, min(18.0, torques[i]))

        try:
            for motor_num in range(1, 7):
                joint_idx = motor_num - 1
                active_kp = self.mit_kp[joint_idx]
                active_pos = positions[joint_idx]
                
                if self.master_slave_enable:
                    if self.slave_follow_mode:
                        active_kp = self.kp_follow[joint_idx]
                        active_pos = self.master_joint_positions[joint_idx]
                    else:
                        active_pos = self.current_joint_positions[joint_idx]

                self.piper.JointMitCtrl(
                    motor_num=motor_num,
                    pos_ref=active_pos, 
                    vel_ref=velocities[joint_idx],
                    kp=active_kp,       
                    kd=self.mit_kd[joint_idx],
                    t_ref=torques[joint_idx],
                )
        except Exception as e:
            rospy.logerr("Error in JointMitCtrl: %s", str(e))

        if self.enable_gripper and self.gripper_exist:
            with self.gripper_cmd_lock:
                raw_cmd_pos = self.gripper_cmd_position
                gripper_eff = self.gripper_cmd_effort

            gripper_target = raw_cmd_pos

            # 接管锁逻辑
            if self.takeover_locked:
                if abs(gripper_target - self.locked_gripper_pos) < 0.005:
                    self.takeover_locked = False
                    rospy.loginfo("[%s] ✅ 夹爪已对齐，成功解除接管锁定！", self.topic_prefix)
                else:
                    gripper_target = self.locked_gripper_pos

            # =========================================================================
            # ✅ “悬停+放大力反馈”模式
            # =========================================================================
            if self.master_slave_enable:
                if self.slave_follow_mode:
                    # 推理模式：锁定跟随
                    with self.control_data_lock:
                        target_pos = self.master_gripper_position
                    if self.gripper_reverse:
                        target_pos = self.gripper_range - target_pos
                    joint_6 = round(target_pos * 1000 * 1000 * self.gripper_val_mutiple)
                    gripper_effort = round(3.0 * 1000)
                else:
                    # 遥操模式
                    with self.control_data_lock:
                        robot_pos = self.master_gripper_position
                        robot_eff = abs(self.master_gripper_effort)

                    force_gain = 2.5      
                    eff_threshold = 0.25  

                    if robot_eff > eff_threshold:
                        # 状态A【夹到硬物】：强制将目标设为Robot当前位置，把你的手顶住
                        target_pos = robot_pos
                        if self.gripper_reverse:
                            target_pos = self.gripper_range - target_pos
                        joint_6 = round(target_pos * 1000 * 1000 * self.gripper_val_mutiple)
                        
                        feedback_effort = robot_eff * force_gain
                        feedback_effort = max(0.5, min(feedback_effort, 3.0))
                        gripper_effort = round(feedback_effort * 1000)
                    else:
                        # 状态B【空气中空抓 / 松开手】：目标设为遥操端【自己的当前物理位置】悬停
                        target_pos = self.current_gripper_pos
                        if self.gripper_reverse:
                            target_pos = self.gripper_range - target_pos
                        joint_6 = round(target_pos * 1000 * 1000 * self.gripper_val_mutiple)
                        
                        gripper_effort = round(0.15 * 1000)
            else:
                # 机器端逻辑
                joint_6 = round(gripper_target * 1000 * 1000 * self.gripper_val_mutiple)
                if self.enable_gripper_haptic:
                    effort_val = -gripper_eff
                    effort_val = max(0, min(effort_val, 0.5))
                    gripper_effort = round(effort_val * 1000)
                else:
                    gripper_effort = round(3 * 1000)
            # =========================================================================

            joint_6 = max(0, min(80000, joint_6))
            if abs(joint_6) < 200:
                joint_6 = 0

            if gripper_effort < 50:
                self.piper.GripperCtrl(abs(joint_6), gripper_effort, 0x00, 0)
            else:
                self.piper.GripperCtrl(abs(joint_6), gripper_effort, 0x01, 0)

    # Publish thread
    def PublishThread(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown() and self.publish_thread_running:
            self.PublishArmState()
            self.PublishArmEndPose()
            self.PublishArmJointAndGripper()
            rate.sleep()

    def PublishArmState(self):
        arm_status = PiperStatusMsg()
        arm_status.ctrl_mode = self.piper.GetArmStatus().arm_status.ctrl_mode
        arm_status.arm_status = self.piper.GetArmStatus().arm_status.arm_status
        arm_status.mode_feedback = self.piper.GetArmStatus().arm_status.mode_feed
        arm_status.teach_status = self.piper.GetArmStatus().arm_status.teach_status
        arm_status.motion_status = self.piper.GetArmStatus().arm_status.motion_status
        arm_status.trajectory_num = self.piper.GetArmStatus().arm_status.trajectory_num
        arm_status.err_code = self.piper.GetArmStatus().arm_status.err_code
        arm_status.joint_1_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        arm_status.joint_2_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        arm_status.joint_3_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        arm_status.joint_4_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        arm_status.joint_5_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        arm_status.joint_6_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        arm_status.communication_status_joint_1 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        )
        arm_status.communication_status_joint_2 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        )
        arm_status.communication_status_joint_3 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        )
        arm_status.communication_status_joint_4 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        )
        arm_status.communication_status_joint_5 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        )
        arm_status.communication_status_joint_6 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_6
        )
        self.arm_status_pub.publish(arm_status)

    def PublishArmJointAndGripper(self):
        joint_0: float = (self.piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
        joint_1: float = (self.piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
        joint_2: float = (self.piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
        joint_3: float = (self.piper.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
        joint_4: float = (self.piper.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
        joint_5: float = (self.piper.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
        vel_0: float = self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000
        vel_1: float = self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000
        vel_2: float = self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000
        vel_3: float = self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000
        vel_4: float = self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000
        vel_5: float = self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000
        effort_0: float = self.piper.GetArmHighSpdInfoMsgs().motor_1.effort / 1000
        effort_1: float = self.piper.GetArmHighSpdInfoMsgs().motor_2.effort / 1000
        effort_2: float = self.piper.GetArmHighSpdInfoMsgs().motor_3.effort / 1000
        effort_3: float = self.piper.GetArmHighSpdInfoMsgs().motor_4.effort / 1000
        effort_4: float = self.piper.GetArmHighSpdInfoMsgs().motor_5.effort / 1000
        effort_5: float = self.piper.GetArmHighSpdInfoMsgs().motor_6.effort / 1000

        with self.control_data_lock:
            self.current_joint_positions = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5]
            self.current_joint_velocities = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5]
            self.current_joint_efforts = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5]

        if self.enable_gripper and self.gripper_exist:
            joint_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
            if self.gripper_reverse:
                joint_6 = self.gripper_range - joint_6
                
            # ✅ 同步更新当前的实际夹爪位置给锁定功能使用
            self.current_gripper_pos = joint_6
                
            effort_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort / 1000
            raw_positions = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
            raw_velocities = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0]
            raw_efforts = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5, effort_6]
        else:
            raw_positions = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5]
            raw_velocities = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5]
            raw_efforts = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5]

        if self.filter_enable:
            with self.filter_lock:
                if not self.filter_initialized:
                    self.filtered_positions = raw_positions[:]
                    self.filtered_velocities = raw_velocities[:]
                    self.filtered_efforts = raw_efforts[:]
                    self.filter_initialized = True
                else:
                    for i in range(len(raw_positions)):
                        self.filtered_positions[i] = (
                            self.filter_alpha_pos * raw_positions[i]
                            + (1.0 - self.filter_alpha_pos) * self.filtered_positions[i]
                        )
                    for i in range(len(raw_velocities)):
                        self.filtered_velocities[i] = (
                            self.filter_alpha_vel * raw_velocities[i]
                            + (1.0 - self.filter_alpha_vel) * self.filtered_velocities[i]
                        )
                    for i in range(len(raw_efforts)):
                        self.filtered_efforts[i] = (
                            self.filter_alpha_effort * raw_efforts[i]
                            + (1.0 - self.filter_alpha_effort) * self.filtered_efforts[i]
                        )
                filtered_positions = self.filtered_positions[:]
                filtered_velocities = self.filtered_velocities[:]
                filtered_efforts = self.filtered_efforts[:]
        else:
            filtered_positions = raw_positions[:]
            filtered_velocities = raw_velocities[:]
            filtered_efforts = raw_efforts[:]

        self.joint_states.header.stamp = rospy.Time.now()
        self.joint_states.position = filtered_positions
        self.joint_states.velocity = filtered_velocities
        self.joint_states.effort = filtered_efforts

        self.joint_pub.publish(self.joint_states)

    def PublishArmEndPose(self):
        endpos = PoseStamped()
        endpos.pose.position.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        endpos.pose.position.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        endpos.pose.position.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        roll = self.piper.GetArmEndPoseMsgs().end_pose.RX_axis / 1000
        pitch = self.piper.GetArmEndPoseMsgs().end_pose.RY_axis / 1000
        yaw = self.piper.GetArmEndPoseMsgs().end_pose.RZ_axis / 1000
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        endpos.pose.orientation.x = quaternion[0]
        endpos.pose.orientation.y = quaternion[1]
        endpos.pose.orientation.z = quaternion[2]
        endpos.pose.orientation.w = quaternion[3]
        endpos.header.stamp = rospy.Time.now()
        self.end_pose_pub.publish(endpos)

        end_pose_euler = PiperEulerPose()
        end_pose_euler.header.stamp = rospy.Time.now()
        end_pose_euler.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        end_pose_euler.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        end_pose_euler.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        end_pose_euler.roll = roll
        end_pose_euler.pitch = pitch
        end_pose_euler.yaw = yaw
        self.end_pose_euler_pub.publish(end_pose_euler)

    def handle_gripper_service(self, req):
        response = GripperResponse()
        response.code = 15999
        response.status = False
        if self.gripper_exist:
            gripper_angle = round(max(0, min(req.gripper_angle, 0.07)) * 1e6)
            gripper_effort = round(max(0.5, min(req.gripper_effort, 2)) * 1e3)
            if req.gripper_code not in [0x00, 0x01, 0x02, 0x03]:
                gripper_code = 1
                response.code = 15901
            else:
                gripper_code = req.gripper_code
            if req.set_zero not in [0x00, 0xAE]:
                set_zero = 0
                response.code = 15902
            else:
                set_zero = req.set_zero
            response.code = 15900
            self.piper.GripperCtrl(abs(gripper_angle), gripper_effort, gripper_code, set_zero)
            response.status = True
        else:
            response.code = 15903
            response.status = False
        return response

    def handle_enable_service(self, req):
        enable_flag = False
        loop_flag = False
        timeout = 5
        start_time = time.time()
        elapsed_time_flag = False
        while not loop_flag:
            elapsed_time = time.time() - start_time
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if req.enable_request:
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
            else:
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x02, 0)
            self.__enable_flag = enable_flag
            if enable_flag == req.enable_request:
                loop_flag = True
                enable_flag = True
            else:
                loop_flag = False
                enable_flag = False
            if elapsed_time > timeout:
                elapsed_time_flag = True
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        response = enable_flag
        return EnableResponse(response)

    def handle_stop_service(self, req):
        response = TriggerResponse()
        response.success = False
        response.message = "stop piper failed"
        self.piper.MotionCtrl_1(0x01, 0, 0)
        response.success = True
        response.message = "stop piper success"
        return response

    def handle_reset_service(self, req):
        response = TriggerResponse()
        response.success = False
        response.message = "reset piper failed"
        self.piper.MotionCtrl_1(0x02, 0, 0)
        response.success = True
        response.message = "reset piper success"
        return response

    def handle_go_zero_service(self, req):
        response = GoZeroResponse()
        response.status = False
        response.code = 151000
        if req.is_mit_mode:
            self.piper.MotionCtrl_2(0x01, 0x01, 50, 0xAD)
        else:
            self.piper.MotionCtrl_2(0x01, 0x01, 50, 0)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        response.status = True
        response.code = 151001
        return response

    def handle_block_arm_service(self, req):
        response = SetBoolResponse()
        if req.data:
            response.success = req.data
            response.message = "You will block arm ctrl msg send"
        else:
            response.success = req.data
            response.message = "You will unblock arm ctrl msg send"
        self.block_ctrl_flag = req.data
        return response

    def EnableArm(self):
        enable_flag = False
        timeout = 5
        start_time = time.time()

        while not enable_flag:
            elapsed_time = time.time() - start_time
            enable_flag = (
                self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            )
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            if enable_flag:
                self.__enable_flag = True

            if elapsed_time > timeout:
                break
            time.sleep(1)

    def Run(self):
        if self.auto_enable:
            enable_thread = threading.Thread(target=self.EnableArm, daemon=True)
            enable_thread.start()

        self.publish_thread_running = True
        self.publish_thread = threading.Thread(target=self.PublishThread, daemon=True)
        self.publish_thread.start()

        self.control_thread_running = True
        self.control_thread = threading.Thread(target=self.ControlThread, daemon=True)
        self.control_thread.start()

        rospy.spin()


if __name__ == "__main__":
    try:
        check_ros_master()
        node = C_PiperCtrlNode()
        node.Run()
    except rospy.ROSInterruptException:
        pass