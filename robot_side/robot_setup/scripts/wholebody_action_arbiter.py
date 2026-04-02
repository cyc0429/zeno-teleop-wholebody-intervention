#!/usr/bin/env python3
"""Whole-body action arbiter for decoupled Arm and Base intervention.

This node keeps policy and teleop streams running in parallel for BOTH arms and base.
It switches the final executed source based on independent Arm and Base modes.
  - Arm Mode:  POLICY or HUMAN (Controls left/right JointStates)
  - Base Mode: POLICY or HUMAN (Controls base Twist)

Output topics are fixed robot command topics.
Intervention flags are published for downstream offline RL data collection.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32MultiArray


MODE_POLICY = "POLICY"
MODE_HUMAN = "HUMAN"
VALID_MODES = {MODE_POLICY, MODE_HUMAN}

SRC_POLICY = "policy"
SRC_HUMAN = "human"


@dataclass
class TimedJointState:
    msg: JointState
    recv_time: rospy.Time

@dataclass
class TimedTwist:
    msg: Twist
    recv_time: rospy.Time


@dataclass
class ModeTransition:
    start_time: rospy.Time
    arm_changed: bool
    base_changed: bool
    start_msg_left: Optional[JointState]
    start_msg_right: Optional[JointState]
    start_msg_base: Optional[Twist]


# ================== Blending Utilities for JointState ==================
def clone_joint_state(msg: JointState, retimestamp: bool, stamp: rospy.Time) -> JointState:
    out = JointState()
    out.header.seq = msg.header.seq
    out.header.frame_id = msg.header.frame_id
    out.header.stamp = stamp if retimestamp else msg.header.stamp
    out.name = list(msg.name)
    out.position = list(msg.position)
    out.velocity = list(msg.velocity)
    out.effort = list(msg.effort)
    return out

def _blend_numeric_seq(start_values, target_values, alpha: float):
    if not start_values or not target_values or len(start_values) != len(target_values):
        return list(target_values) if target_values else []
    w0 = 1.0 - alpha
    return [w0 * float(s) + alpha * float(t) for s, t in zip(start_values, target_values)]

def blend_joint_state(start_msg: JointState, target_msg: JointState, alpha: float, retimestamp: bool, stamp: rospy.Time) -> JointState:
    if alpha <= 0.0: return clone_joint_state(start_msg, retimestamp, stamp)
    if alpha >= 1.0: return clone_joint_state(target_msg, retimestamp, stamp)
    out = clone_joint_state(target_msg, retimestamp, stamp)
    out.position = _blend_numeric_seq(start_msg.position, target_msg.position, alpha)
    out.velocity = _blend_numeric_seq(start_msg.velocity, target_msg.velocity, alpha)
    out.effort = _blend_numeric_seq(start_msg.effort, target_msg.effort, alpha)
    return out


# ================== Blending Utilities for Twist (Base) ==================
def clone_twist(msg: Twist) -> Twist:
    out = Twist()
    out.linear.x, out.linear.y, out.linear.z = msg.linear.x, msg.linear.y, msg.linear.z
    out.angular.x, out.angular.y, out.angular.z = msg.angular.x, msg.angular.y, msg.angular.z
    return out

def blend_twist(start_msg: Twist, target_msg: Twist, alpha: float) -> Twist:
    if alpha <= 0.0: return clone_twist(start_msg)
    if alpha >= 1.0: return clone_twist(target_msg)
    out = Twist()
    w0 = 1.0 - alpha
    out.linear.x = w0 * start_msg.linear.x + alpha * target_msg.linear.x
    out.linear.y = w0 * start_msg.linear.y + alpha * target_msg.linear.y
    out.linear.z = w0 * start_msg.linear.z + alpha * target_msg.linear.z
    out.angular.x = w0 * start_msg.angular.x + alpha * target_msg.angular.x
    out.angular.y = w0 * start_msg.angular.y + alpha * target_msg.angular.y
    out.angular.z = w0 * start_msg.angular.z + alpha * target_msg.angular.z
    return out


class WholebodyActionArbiter:
    def __init__(self) -> None:
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 200.0))
        self.selected_source_timeout_sec = float(rospy.get_param("~selected_source_timeout_sec", 0.30))
        self.fallback_source_timeout_sec = float(rospy.get_param("~fallback_source_timeout_sec", 0.30))
        self.hold_last_on_timeout = bool(rospy.get_param("~hold_last_on_timeout", True))
        self.allow_fallback_to_other_source = bool(rospy.get_param("~allow_fallback_to_other_source", False))
        self.retimestamp_output = bool(rospy.get_param("~retimestamp_output", True))
        
        # ✅ 【核心修复1】：独立拆分融合时间！手臂平滑防抖(0.6s)，底盘极速刹车(0.1s)！
        legacy_blend = float(rospy.get_param("~mode_switch_blend_sec", 0.60))
        self.arm_mode_switch_blend_sec = max(0.0, float(rospy.get_param("~arm_mode_switch_blend_sec", legacy_blend)))
        self.base_mode_switch_blend_sec = max(0.0, float(rospy.get_param("~base_mode_switch_blend_sec", 0.10)))

        # Topics setup
        self.left_policy_topic = str(rospy.get_param("~left_policy_topic", "/robot/arm_left/vla_joint_cmd"))
        self.right_policy_topic = str(rospy.get_param("~right_policy_topic", "/robot/arm_right/vla_joint_cmd"))
        self.base_policy_topic = str(rospy.get_param("~base_policy_topic", "/robot/base/vla_cmd_vel"))
        
        self.left_human_topic = str(rospy.get_param("~left_human_topic", "/teleop/arm_left/joint_states"))
        self.right_human_topic = str(rospy.get_param("~right_human_topic", "/teleop/arm_right/joint_states"))
        self.base_human_topic = str(rospy.get_param("~base_human_topic", "/teleop/base/cmd_vel"))

        self.left_output_topic = str(rospy.get_param("~left_output_topic", "/robot/arm_left/robot_cmd"))
        self.right_output_topic = str(rospy.get_param("~right_output_topic", "/robot/arm_right/robot_cmd"))
        self.base_output_topic = str(rospy.get_param("~base_output_topic", "/cmd_vel"))

        self.mode_cmd_topic = str(rospy.get_param("~mode_cmd_topic", "/intervention/mode_cmd"))
        self.flags_topic = str(rospy.get_param("~flags_topic", "/intervention/flags"))

        # Orthogonal State Machine
        self._lock = threading.Lock()
        self._arm_mode = MODE_HUMAN
        self._base_mode = MODE_HUMAN
        
        # Data Caches
        self._latest_joints: Dict[Tuple[str, str], Optional[TimedJointState]] = {
            ("left", SRC_POLICY): None, ("left", SRC_HUMAN): None,
            ("right", SRC_POLICY): None, ("right", SRC_HUMAN): None,
        }
        self._latest_twists: Dict[str, Optional[TimedTwist]] = {
            SRC_POLICY: None, SRC_HUMAN: None
        }
        
        self._last_pub_left: Optional[TimedJointState] = None
        self._last_pub_right: Optional[TimedJointState] = None
        self._last_pub_base: Optional[TimedTwist] = None

        self._mode_transition: Optional[ModeTransition] = None

        # Publishers
        self.left_pub = rospy.Publisher(self.left_output_topic, JointState, queue_size=1)
        self.right_pub = rospy.Publisher(self.right_output_topic, JointState, queue_size=1)
        self.base_pub = rospy.Publisher(self.base_output_topic, Twist, queue_size=1)
        self.flags_pub = rospy.Publisher(self.flags_topic, Float32MultiArray, queue_size=1, latch=True)

        # Subscribers
        rospy.Subscriber(self.left_policy_topic, JointState, self._joint_cb, ("left", SRC_POLICY), queue_size=1)
        rospy.Subscriber(self.right_policy_topic, JointState, self._joint_cb, ("right", SRC_POLICY), queue_size=1)
        rospy.Subscriber(self.left_human_topic, JointState, self._joint_cb, ("left", SRC_HUMAN), queue_size=1)
        rospy.Subscriber(self.right_human_topic, JointState, self._joint_cb, ("right", SRC_HUMAN), queue_size=1)
        
        rospy.Subscriber(self.base_policy_topic, Twist, self._twist_cb, SRC_POLICY, queue_size=1)
        rospy.Subscriber(self.base_human_topic, Twist, self._twist_cb, SRC_HUMAN, queue_size=1)

        rospy.Subscriber(self.mode_cmd_topic, String, self._mode_cmd_cb, queue_size=10)

        self._publish_flags()
        rospy.loginfo("[Arbiter] Whole-body arbiter started. Rate=%.1fHz", self.publish_rate_hz)

    def _joint_cb(self, msg: JointState, args: Tuple[str, str]) -> None:
        arm, source = args
        with self._lock:
            self._latest_joints[(arm, source)] = TimedJointState(msg, rospy.Time.now())

    def _twist_cb(self, msg: Twist, source: str) -> None:
        with self._lock:
            self._latest_twists[source] = TimedTwist(msg, rospy.Time.now())

    def _set_modes(self, new_arm: str, new_base: str, reason: str) -> None:
        with self._lock:
            arm_changed = (new_arm != self._arm_mode)
            base_changed = (new_base != self._base_mode)
            
            self._arm_mode = new_arm
            self._base_mode = new_base
            
            if arm_changed or base_changed:
                self._mode_transition = None
                if self.arm_mode_switch_blend_sec > 0.0 or self.base_mode_switch_blend_sec > 0.0:
                    self._mode_transition = ModeTransition(
                        start_time=rospy.Time.now(),
                        arm_changed=arm_changed,
                        base_changed=base_changed,
                        start_msg_left=clone_joint_state(self._last_pub_left.msg, False, rospy.Time(0)) if self._last_pub_left else None,
                        start_msg_right=clone_joint_state(self._last_pub_right.msg, False, rospy.Time(0)) if self._last_pub_right else None,
                        start_msg_base=clone_twist(self._last_pub_base.msg) if self._last_pub_base else None
                    )
                rospy.loginfo("[Arbiter] Switch -> Arm:%s Base:%s (reason: %s)", new_arm, new_base, reason)
                
        # 只要有信号，强制全网广播 Flag
        self._publish_flags()

    def _mode_cmd_cb(self, msg: String) -> None:
        cmd = msg.data.strip().lower()
        with self._lock:
            new_arm, new_base = self._arm_mode, self._base_mode
            # ✅ 【核心修复2】：强力兼容你的自定义按键 h, w, p, a, b
            if cmd in ("whole_human"):
                new_arm, new_base = MODE_HUMAN, MODE_HUMAN
            elif cmd in ("all_policy"):
                new_arm, new_base = MODE_POLICY, MODE_POLICY
            elif cmd in ("toggle_arm"):
                new_arm = MODE_HUMAN if self._arm_mode == MODE_POLICY else MODE_POLICY
            elif cmd in ("toggle_base"):
                new_base = MODE_HUMAN if self._base_mode == MODE_POLICY else MODE_POLICY
            else:
                return
        self._set_modes(new_arm, new_base, f"cmd:{cmd}")

    def _publish_flags(self) -> None:
        with self._lock:
            arm_flag = 1.0 if self._arm_mode == MODE_HUMAN else 0.0
            base_flag = 1.0 if self._base_mode == MODE_HUMAN else 0.0
        
        msg = Float32MultiArray()
        msg.data = [arm_flag, base_flag]
        self.flags_pub.publish(msg)

    # --- Selection Logic ---
    def _is_fresh(self, recv_time: rospy.Time, now: rospy.Time, timeout: float) -> bool:
        if timeout < 0.0: return True
        return (now - recv_time).to_sec() <= timeout

    def _pick_joint(self, arm: str, now: rospy.Time) -> Tuple[Optional[JointState], str]:
        with self._lock:
            mode = self._arm_mode
            primary = SRC_HUMAN if mode == MODE_HUMAN else SRC_POLICY
            secondary = SRC_POLICY if primary == SRC_HUMAN else SRC_HUMAN
            p_state = self._latest_joints[(arm, primary)]
            s_state = self._latest_joints[(arm, secondary)]
            hold_state = self._last_pub_left if arm == "left" else self._last_pub_right

        if p_state and self._is_fresh(p_state.recv_time, now, self.selected_source_timeout_sec):
            return clone_joint_state(p_state.msg, self.retimestamp_output, now), primary
        if self.allow_fallback_to_other_source and s_state and self._is_fresh(s_state.recv_time, now, self.fallback_source_timeout_sec):
            return clone_joint_state(s_state.msg, self.retimestamp_output, now), f"{secondary}_fb"
        
        # 手臂可以使用 hold 逻辑（停在原地）
        if self.hold_last_on_timeout and hold_state:
            return clone_joint_state(hold_state.msg, self.retimestamp_output, now), "hold"
        return None, "none"

    def _pick_twist(self, now: rospy.Time) -> Tuple[Optional[Twist], str]:
        with self._lock:
            mode = self._base_mode
            primary = SRC_HUMAN if mode == MODE_HUMAN else SRC_POLICY
            p_state = self._latest_twists[primary]

        if p_state and self._is_fresh(p_state.recv_time, now, self.selected_source_timeout_sec):
            return clone_twist(p_state.msg), primary
            
        # ✅ 【核心修复3：底盘绝对保命机制】
        # 不管是 AI 挂了，还是你没踩踏板，只要没收到新鲜指令，底盘必须立刻发 0 刹车！
        # 绝对不能去 hold（维持上一帧速度）或者 fallback！
        return Twist(), f"{primary}_emergency_stop"

    def _run_once(self) -> None:
        now = rospy.Time.now()
        
        # 1. Selection
        left_msg, _ = self._pick_joint("left", now)
        right_msg, _ = self._pick_joint("right", now)
        base_msg, _ = self._pick_twist(now)

        # 2. Transition Blending Calculation
        with self._lock:
            transition = self._mode_transition
        
        alpha_arm = 1.0
        alpha_base = 1.0
        if transition:
            elapsed = (now - transition.start_time).to_sec()
            
            # 当两个部位的融合时间都走完时，清除 Transition 状态
            if elapsed >= max(self.arm_mode_switch_blend_sec, self.base_mode_switch_blend_sec):
                with self._lock:
                    if self._mode_transition is transition:
                        self._mode_transition = None
                        
            # 独立计算融合率
            if self.arm_mode_switch_blend_sec > 0.0:
                alpha_arm = max(0.0, min(1.0, elapsed / self.arm_mode_switch_blend_sec))
            if self.base_mode_switch_blend_sec > 0.0:
                alpha_base = max(0.0, min(1.0, elapsed / self.base_mode_switch_blend_sec))

        # 3. Apply Blending selectively based on what changed
        if transition:
            if transition.arm_changed and alpha_arm < 1.0:
                if left_msg and transition.start_msg_left:
                    left_msg = blend_joint_state(transition.start_msg_left, left_msg, alpha_arm, self.retimestamp_output, now)
                if right_msg and transition.start_msg_right:
                    right_msg = blend_joint_state(transition.start_msg_right, right_msg, alpha_arm, self.retimestamp_output, now)
            
            if transition.base_changed and alpha_base < 1.0:
                if base_msg and transition.start_msg_base:
                    base_msg = blend_twist(transition.start_msg_base, base_msg, alpha_base)

        # 4. Publish & Cache
        if left_msg:
            self.left_pub.publish(left_msg)
            with self._lock: self._last_pub_left = TimedJointState(left_msg, now)
            
        if right_msg:
            self.right_pub.publish(right_msg)
            with self._lock: self._last_pub_right = TimedJointState(right_msg, now)
            
        if base_msg:
            self.base_pub.publish(base_msg)
            with self._lock: self._last_pub_base = TimedTwist(base_msg, now)

    def spin(self) -> None:
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown():
            self._run_once()
            rate.sleep()

def main() -> None:
    rospy.init_node("wholebody_action_arbiter")
    WholebodyActionArbiter().spin()

if __name__ == "__main__":
    main()