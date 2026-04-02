#!/usr/bin/env python3
"""Repeat A/B joint targets and base velocities to fake a whole-body policy command stream.

Example:
  python3 fake_wholebody_policy.py

Optional custom parameters:
  python3 fake_wholebody_policy.py \
    --left-waypoint-a "0.10,0.20,0,0,0,0,0.02" \
    --left-waypoint-b "0.20,0.10,0,0,0,0,0.04" \
    --base-speed 0.1 \
    --base-turn-speed 0.5 \
    --base-duration 3.0
"""

from __future__ import annotations

import argparse
import ast
import math
from typing import List

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist


DEFAULT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
# Slightly larger default motion than the original demo path.
# 关节顺序: [底座旋转, 大臂俯仰, 小臂俯仰(负数!), 腕部翻滚, 腕部俯仰, 腕部旋转, 夹爪开合(米)]
# A 点：轻微抬起并向左转，夹爪微开 (20mm)
DEFAULT_LEFT_A = [0.20, 0.20, -0.50, 0.00, 0.30, 0.00, 0.02]

# B 点：向右转，大臂放下，小臂折叠更深，夹爪张大 (60mm)
DEFAULT_LEFT_B = [-0.20, 0.40, -0.80, 0.00, -0.20, 0.00, 0.06]


def parse_waypoint(text: str, expected_len: int, arg_name: str) -> List[float]:
    """Parse waypoint from '[..]' or 'v1,v2,...' format."""
    raw = text.strip()
    if not raw:
        raise argparse.ArgumentTypeError(f"{arg_name} must not be empty")

    try:
        if raw.startswith("["):
            values = ast.literal_eval(raw)
        else:
            values = [v.strip() for v in raw.split(",")]
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"{arg_name} parse failed: {exc}") from exc

    if not isinstance(values, (list, tuple)):
        raise argparse.ArgumentTypeError(f"{arg_name} must be a list/tuple or comma-separated values")

    try:
        out = [float(v) for v in values]
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"{arg_name} contains non-numeric value: {exc}") from exc

    if len(out) != expected_len:
        raise argparse.ArgumentTypeError(f"{arg_name} length must be {expected_len}, got {len(out)}")

    return out


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Arm Arguments
    parser.add_argument("--left-topic", default="/robot/arm_left/vla_joint_cmd")
    parser.add_argument("--right-topic", default="/robot/arm_right/vla_joint_cmd")
    parser.add_argument("--disable-left", action="store_true", help="Do not publish left arm command")
    parser.add_argument("--disable-right", action="store_true", help="Do not publish right arm command")
    
    # Base Arguments
    parser.add_argument("--base-topic", default="/robot/base/vla_cmd_vel")
    parser.add_argument("--disable-base", action="store_true", help="Do not publish base command")
    parser.add_argument("--base-speed", type=float, default=0.05, help="Linear x speed for base (m/s)")
    parser.add_argument("--base-turn-speed", type=float, default=0.3, help="Angular z speed for base (rad/s)")
    parser.add_argument("--base-duration", type=float, default=2.0, help="Time to move in one direction before switching (seconds)")

    # Global Arguments
    parser.add_argument("--rate", type=float, default=100.0, help="Publish rate (Hz)")
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=2.0,
        help="A->B travel time (seconds) for smooth cosine replay (Arms)",
    )
    
    # Waypoints
    parser.add_argument(
        "--names",
        default=",".join(DEFAULT_NAMES),
        help="Joint names, comma-separated. Must match waypoint length.",
    )
    parser.add_argument(
        "--left-waypoint-a",
        default=",".join(str(v) for v in DEFAULT_LEFT_A),
        help="Left arm waypoint A, list or comma-separated floats",
    )
    parser.add_argument(
        "--left-waypoint-b",
        default=",".join(str(v) for v in DEFAULT_LEFT_B),
        help="Left arm waypoint B, list or comma-separated floats",
    )
    parser.add_argument(
        "--right-waypoint-a",
        default="",
        help="Right arm waypoint A, if empty uses left-waypoint-a",
    )
    parser.add_argument(
        "--right-waypoint-b",
        default="",
        help="Right arm waypoint B, if empty uses left-waypoint-b",
    )
    return parser


def parse_joint_names(names_arg: str) -> List[str]:
    names = [s.strip() for s in names_arg.split(",") if s.strip()]
    if not names:
        raise ValueError("--names must contain at least one joint name")
    return names


def build_joint_state(names: List[str], position: List[float], stamp: rospy.Time) -> JointState:
    msg = JointState()
    msg.header.stamp = stamp
    msg.name = names
    msg.position = position
    msg.velocity = [0.0] * len(names)
    msg.effort = [0.0] * len(names)
    return msg


def interpolate(a: List[float], b: List[float], alpha: float) -> List[float]:
    return [av + alpha * (bv - av) for av, bv in zip(a, b)]


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.rate <= 0.0:
        parser.error("--rate must be > 0")
    if args.hold_sec <= 0.0:
        parser.error("--hold-sec must be > 0")
    if args.disable_left and args.disable_right and args.disable_base:
        parser.error("All outputs (left, right, base) are disabled. Enable at least one publisher.")

    names = parse_joint_names(args.names)
    n = len(names)

    left_a = parse_waypoint(args.left_waypoint_a, n, "--left-waypoint-a")
    left_b = parse_waypoint(args.left_waypoint_b, n, "--left-waypoint-b")
    right_a = parse_waypoint(args.right_waypoint_a, n, "--right-waypoint-a") if args.right_waypoint_a else left_a
    right_b = parse_waypoint(args.right_waypoint_b, n, "--right-waypoint-b") if args.right_waypoint_b else left_b

    rospy.init_node("fake_wholebody_policy_loop")

    # Setup Publishers
    pub_left = None if args.disable_left else rospy.Publisher(args.left_topic, JointState, queue_size=1)
    pub_right = None if args.disable_right else rospy.Publisher(args.right_topic, JointState, queue_size=1)
    pub_base = None if args.disable_base else rospy.Publisher(args.base_topic, Twist, queue_size=1)

    rospy.loginfo("🧠 [Mock AI] Fake whole-body policy loop started")
    
    # Log Arm config
    if pub_left is not None or pub_right is not None:
        rospy.loginfo("--- Arm Configuration ---")
    if pub_left is not None:
        rospy.loginfo("  left topic: %s", args.left_topic)
        rospy.loginfo("  left A: %s", [round(v, 4) for v in left_a])
        rospy.loginfo("  left B: %s", [round(v, 4) for v in left_b])
    if pub_right is not None:
        rospy.loginfo("  right topic: %s", args.right_topic)
        rospy.loginfo("  right A: %s", [round(v, 4) for v in right_a])
        rospy.loginfo("  right B: %s", [round(v, 4) for v in right_b])
        
    # Log Base config
    if pub_base is not None:
        rospy.loginfo("--- Base Configuration ---")
        rospy.loginfo("  base topic: %s", args.base_topic)
        rospy.loginfo("  linear speed: %.3f m/s", args.base_speed)
        rospy.loginfo("  angular speed: %.3f rad/s", args.base_turn_speed)
        rospy.loginfo("  switch every: %.1f sec", args.base_duration)

    rospy.loginfo("--- System ---")
    rospy.loginfo("  publish rate: %.2f Hz", args.rate)

    rate = rospy.Rate(args.rate)
    
    # Wait for ROS time to initialize (handles edge cases in simulation)
    while rospy.Time.now().to_sec() == 0 and not rospy.is_shutdown():
        rate.sleep()

    start_time = rospy.Time.now()
    
    # Base state tracking (0: Forward, 1: Backward, 2: Turn Left, 3: Turn Right)
    base_state = 0
    base_state_names = ["Forward", "Backward", "Turn Left", "Turn Right"]
    base_last_switch_time = start_time

    rospy.loginfo("🚀 Publishing combined streams...")

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        
        # ---------------------------------------------------------
        # 1. Arm Logic (Continuous Cosine Interpolation)
        # ---------------------------------------------------------
        if pub_left is not None or pub_right is not None:
            # alpha in [0, 1], periodic and C1-smooth at A/B endpoints.
            elapsed = max(0.0, (now - start_time).to_sec())
            alpha = 0.5 * (1.0 - math.cos(math.pi * elapsed / args.hold_sec))
            
            if pub_left is not None:
                left_pos = interpolate(left_a, left_b, alpha)
                pub_left.publish(build_joint_state(names, left_pos, now))
            
            if pub_right is not None:
                right_pos = interpolate(right_a, right_b, alpha)
                pub_right.publish(build_joint_state(names, right_pos, now))

        # ---------------------------------------------------------
        # 2. Base Logic (4-step State Machine)
        # ---------------------------------------------------------
        if pub_base is not None:
            if (now - base_last_switch_time).to_sec() > args.base_duration:
                # 切换到下一个状态
                base_state = (base_state + 1) % 4
                base_last_switch_time = now
                rospy.loginfo(f"🤖 [Mock AI Base] Switched state -> {base_state_names[base_state]}")
            
            base_cmd = Twist()
            
            if base_state == 0:
                # 前进
                base_cmd.linear.x = args.base_speed
            elif base_state == 1:
                # 后退 (回到起始位置)
                base_cmd.linear.x = -args.base_speed
            elif base_state == 2:
                # 原地左转
                base_cmd.angular.z = args.base_turn_speed
            elif base_state == 3:
                # 原地右转 (回到起始朝向)
                base_cmd.angular.z = -args.base_turn_speed
                
            pub_base.publish(base_cmd)

        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass