#!/usr/bin/env python3
"""Keyboard helper for Whole-body (Arm + Base) intervention mode switching.

Controls (H/A/B/P/S/Q; upper or lower case):
  H -> whole_human (Arm HUMAN, Base HUMAN)
  P -> all_policy (Arm POLICY, Base POLICY)
  A -> toggle_arm
  B -> toggle_base
  S -> print status; Q or Ctrl+C -> quit
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from typing import Optional

import rospy
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


class ModeKeyboard:
    def __init__(self) -> None:
        self.mode_cmd_topic = str(rospy.get_param("~mode_cmd_topic", "/intervention/mode_cmd"))
        self.flags_topic = str(rospy.get_param("~flags_topic", "/intervention/flags"))
        
        self.debounce_sec = max(0.0, float(rospy.get_param("~debounce_sec", 0.15)))
        self.status_interval_sec = max(0.1, float(rospy.get_param("~status_interval_sec", 5.0)))
        self.help_topic_timeout_sec = max(
            0.1, float(rospy.get_param("~help_topic_timeout_sec", 3.0))
        )
        self.concise_status = bool(rospy.get_param("~concise_status", True))
        
        self.sync_slave_follow_flag = bool(rospy.get_param("~sync_slave_follow_flag", True))
        
        # 确保指向 teleop，让遥操端变硬
        self.slave_follow_flag_topic = str(
            rospy.get_param("~slave_follow_flag_topic", "/teleop/slave_follow_flag")
        )
        self.policy_mode_follow_value = bool(rospy.get_param("~policy_mode_follow_value", True))
        self.human_mode_follow_value = bool(rospy.get_param("~human_mode_follow_value", False))
        self.follow_publish_interval_sec = max(
            0.05, float(rospy.get_param("~follow_publish_interval_sec", 0.5))
        )

        self._arm_mode = "UNKNOWN"
        self._base_mode = "UNKNOWN"
        self._follow_state = self.human_mode_follow_value
        self._last_key_ts = 0.0
        self._last_status_ts = 0.0
        self._last_follow_pub_ts = 0.0
        self._uncertainty_score: Optional[float] = None
        self._uncertainty_score_base: Optional[float] = None
        self._uncertainty_score_arm: Optional[float] = None
        self._intervention_request: Optional[bool] = None
        self._intervention_request_base: Optional[bool] = None
        self._intervention_request_arm: Optional[bool] = None
        self._last_help_topic_rx_ts = 0.0

        self.uncertainty_topic = str(
            rospy.get_param("~uncertainty_topic", "/safety/uncertainty_score")
        )
        self.uncertainty_topic_base = str(
            rospy.get_param("~uncertainty_topic_base", "/safety/uncertainty_score_base")
        )
        self.uncertainty_topic_arm = str(
            rospy.get_param("~uncertainty_topic_arm", "/safety/uncertainty_score_arm")
        )
        self.intervention_request_topic = str(
            rospy.get_param("~intervention_request_topic", "/safety/intervention_request")
        )
        self.intervention_request_topic_base = str(
            rospy.get_param("~intervention_request_topic_base", "/safety/intervention_request_base")
        )
        self.intervention_request_topic_arm = str(
            rospy.get_param("~intervention_request_topic_arm", "/safety/intervention_request_arm")
        )

        self.mode_cmd_pub = rospy.Publisher(self.mode_cmd_topic, String, queue_size=10)
        self.slave_follow_flag_pub: Optional[rospy.Publisher] = None
        if self.sync_slave_follow_flag:
            self.slave_follow_flag_pub = rospy.Publisher(
                self.slave_follow_flag_topic, Bool, queue_size=10, latch=True
            )
            self.slave_follow_flag_pub.publish(Bool(data=self._follow_state))
            
        rospy.Subscriber(self.flags_topic, Float32MultiArray, self._flags_callback, queue_size=10, tcp_nodelay=True)
        rospy.Subscriber(
            self.uncertainty_topic,
            Float32,
            self._uncertainty_callback,
            queue_size=10,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.uncertainty_topic_base,
            Float32,
            self._uncertainty_base_callback,
            queue_size=10,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.uncertainty_topic_arm,
            Float32,
            self._uncertainty_arm_callback,
            queue_size=10,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.intervention_request_topic,
            Bool,
            self._intervention_request_callback,
            queue_size=10,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.intervention_request_topic_base,
            Bool,
            self._intervention_request_base_callback,
            queue_size=10,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.intervention_request_topic_arm,
            Bool,
            self._intervention_request_arm_callback,
            queue_size=10,
            tcp_nodelay=True,
        )

    def _uncertainty_callback(self, msg: Float32) -> None:
        self._uncertainty_score = float(msg.data)
        self._last_help_topic_rx_ts = time.monotonic()

    def _intervention_request_callback(self, msg: Bool) -> None:
        self._intervention_request = bool(msg.data)
        self._last_help_topic_rx_ts = time.monotonic()

    def _uncertainty_base_callback(self, msg: Float32) -> None:
        self._uncertainty_score_base = float(msg.data)
        self._last_help_topic_rx_ts = time.monotonic()

    def _uncertainty_arm_callback(self, msg: Float32) -> None:
        self._uncertainty_score_arm = float(msg.data)
        self._last_help_topic_rx_ts = time.monotonic()

    def _intervention_request_base_callback(self, msg: Bool) -> None:
        self._intervention_request_base = bool(msg.data)
        self._last_help_topic_rx_ts = time.monotonic()

    def _intervention_request_arm_callback(self, msg: Bool) -> None:
        self._intervention_request_arm = bool(msg.data)
        self._last_help_topic_rx_ts = time.monotonic()

    def _help_status_text(self) -> str:
        now = time.monotonic()
        if (now - self._last_help_topic_rx_ts) > self.help_topic_timeout_sec:
            return "求助功能未开启"
        sf = "N/A" if self._uncertainty_score is None else f"{self._uncertainty_score:.4f}"
        sb = "N/A" if self._uncertainty_score_base is None else f"{self._uncertainty_score_base:.4f}"
        sa = "N/A" if self._uncertainty_score_arm is None else f"{self._uncertainty_score_arm:.4f}"
        rf = "ON" if self._intervention_request else "OFF"
        rb = "ON" if self._intervention_request_base else "OFF"
        ra = "ON" if self._intervention_request_arm else "OFF"
        return f"REQ[F/B/A]={rf}/{rb}/{ra} | U[F/B/A]={sf}/{sb}/{sa}"

    def _publish_follow_for_mode(self, arm_mode: str) -> None:
        if self.slave_follow_flag_pub is None:
            return
        if arm_mode == "POLICY":
            follow = self.policy_mode_follow_value
        elif arm_mode == "HUMAN":
            follow = self.human_mode_follow_value
        else:
            return
        self._follow_state = follow
        self.slave_follow_flag_pub.publish(Bool(data=follow))
        self._last_follow_pub_ts = time.monotonic()

    def _flags_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            return
            
        new_arm = "HUMAN" if msg.data[0] > 0.5 else "POLICY"
        new_base = "HUMAN" if msg.data[1] > 0.5 else "POLICY"
        
        changed = (new_arm != self._arm_mode) or (new_base != self._base_mode)
        
        if changed:
            self._arm_mode = new_arm
            self._base_mode = new_base
            if not self.concise_status:
                print(f"\r[STATUS] Arm: {self._arm_mode:<6} | Base: {self._base_mode:<6}\n", end="")
            self._publish_follow_for_mode(self._arm_mode)

    def _send_mode_cmd(self, cmd: str) -> None:
        self.mode_cmd_pub.publish(String(data=cmd))
        print(f"\r[CMD Sent] -> '{cmd}'\n", end="")

    def _print_help(self) -> None:
        print("\n=============================================")
        print("    全身干预控制台 (Whole-body Intervention)   ")
        print("=============================================")
        print(f" Cmd Topic:   {self.mode_cmd_topic}")
        print(f" Flags Topic: {self.flags_topic}")
        print(f" Slave Flag:  {self.slave_follow_flag_topic}")
        print(f" Help Req:    {self.intervention_request_topic}")
        print(f" Uncertainty: {self.uncertainty_topic}")
        print(f" Help Req B/A:{self.intervention_request_topic_base} | {self.intervention_request_topic_arm}")
        print(f" Uncrt B/A:   {self.uncertainty_topic_base} | {self.uncertainty_topic_arm}")
        print("---------------------------------------------")
        print(" 快捷键绑定:")
        print("   [H] : 全身接管 (Arm->HUMAN, Base->HUMAN)")
        print("   [P] : 一键全自动 (Arm->POLICY, Base->POLICY)")
        print("   [A] : 仅切换手臂 (Toggle Arm)")
        print("   [B] : 仅切换底盘 (Toggle Base)")
        print("   [S] : 打印当前状态")
        print("   [Q] : 退出")
        print("=============================================\n")

    # ✅ 完美复刻官方 teleop_twist_keyboard 的安全读取法
    def _get_key(self, settings, timeout=0.2):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def run(self) -> None:
        self._print_help()
        # 记录初始终端状态
        settings = termios.tcgetattr(sys.stdin)
        
        try:
            while not rospy.is_shutdown():
                now = time.monotonic()
                
                # 心跳打印
                if now - self._last_status_ts >= self.status_interval_sec:
                    self._last_status_ts = now
                    if self._arm_mode != "UNKNOWN":
                        line = (
                            f"[Heartbeat] Arm:{self._arm_mode:<6} Base:{self._base_mode:<6} | "
                            f"{self._help_status_text()}"
                        )
                        if self.concise_status:
                            print("\r" + line.ljust(140), end="", flush=True)
                        else:
                            print(f"\r{line}\n", end="")
                        
                # 同步硬件底层
                if (
                    self.slave_follow_flag_pub is not None
                    and now - self._last_follow_pub_ts >= self.follow_publish_interval_sec
                ):
                    self.slave_follow_flag_pub.publish(Bool(data=self._follow_state))
                    self._last_follow_pub_ts = now

                # ✅ 安全获取按键，如果没按，瞬间恢复终端状态
                key = self._get_key(settings, 0.2)
                
                if not key:
                    continue

                now = time.monotonic()
                if now - self._last_key_ts < self.debounce_sec:
                    continue
                self._last_key_ts = now

                # 键盘映射逻辑
                if key in ("h", "H"):
                    self._send_mode_cmd("whole_human")
                elif key in ("p", "P"):
                    self._send_mode_cmd("all_policy")
                elif key in ("a", "A"):
                    self._send_mode_cmd("toggle_arm")
                elif key in ("b", "B"):
                    self._send_mode_cmd("toggle_base")
                elif key in ("s", "S"):
                    print(
                        f"\r[Current] Arm: {self._arm_mode:<6} | Base: {self._base_mode:<6} | "
                        f"{self._help_status_text()}\n",
                        end="",
                    )
                elif key == '\x03' or key in ("q", "Q"):  # 支持 Ctrl+C 直接退出
                    print("\r[Exit] 退出干预控制台。\n", end="")
                    break

        except Exception as e:
            print(f"\r[Error] {e}")
        finally:
            # 无论发生什么崩溃，保证终端不会死锁
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main() -> None:
    rospy.init_node("intervention_mode_keyboard", anonymous=True)
    ModeKeyboard().run()


if __name__ == "__main__":
    main()