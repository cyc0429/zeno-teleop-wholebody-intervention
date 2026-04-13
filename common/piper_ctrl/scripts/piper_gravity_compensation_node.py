#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gravity compensation torque calculation node for dual-arm robot
Subscribes to joint angles from both arms, computes and publishes gravity compensation torques
"""

import rospy
import numpy as np
import pinocchio as pin
import os
import subprocess
import re
import tempfile
from sensor_msgs.msg import JointState


class GravityCompensationArm:
    """Gravity compensation node for a single arm"""

    def __init__(self, arm_side: str = "left"):
        if arm_side not in ["left", "right"]:
            raise ValueError("arm_side must be 'left' or 'right'")

        self.arm_side = arm_side
        self.name_id = f"[robot_{self.arm_side}]"

        try:
            package_path = subprocess.check_output("rospack find piper_description", shell=True).strip().decode("utf-8")
            urdf_path = os.path.join(package_path, "urdf", "piper_description.urdf")
            
            with open(urdf_path, "r") as f:
                urdf_content = f.read()

            def resolve_package_path(match):
                package_name = match.group(1)
                relative_path = match.group(2)
                try:
                    pkg_path = subprocess.check_output(["rospack", "find", package_name], stderr=subprocess.DEVNULL).strip().decode("utf-8")
                    return os.path.join(pkg_path, relative_path)
                except Exception:
                    return match.group(0)

            urdf_content = re.sub(r"package://([^/]+)/(.+)", resolve_package_path, urdf_content)
            temp_urdf_file = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
            temp_urdf_file.write(urdf_content)
            temp_urdf_file.close()
            processed_urdf_path = temp_urdf_file.name

        except Exception as e:
            rospy.logerr("%s Failed to process URDF file: %s", self.name_id, str(e))
            raise

        try:
            self.robot = pin.RobotWrapper.BuildFromURDF(processed_urdf_path)
            os.unlink(processed_urdf_path)
        except Exception as e:
            rospy.logerr("%s Failed to load robot model: %s", self.name_id, str(e))
            try:
                os.unlink(processed_urdf_path)
            except Exception:
                pass
            raise

        self.data = self.robot.model.createData()

        sub_topic = f"/robot/arm_{arm_side}/joint_states_single"
        pub_topic = f"/robot/arm_{arm_side}/joint_states_compensated"

        self.joint_state_sub = rospy.Subscriber(sub_topic, JointState, self.joint_state_callback, queue_size=1)
        self.torque_pub = rospy.Publisher(pub_topic, JointState, queue_size=1)

        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        
        rospy.loginfo("%s Initialized. Sub: %s -> Pub: %s", self.name_id, sub_topic, pub_topic)

    def joint_state_callback(self, msg):
        try:
            if len(msg.position) < 6:
                return

            if len(msg.position) >= 7:
                joint_positions = np.array(msg.position[:7])
            else:
                joint_positions = np.array(list(msg.position[:6]) + [0.0])

            nq_model = self.robot.model.nq
            if len(joint_positions) < nq_model:
                joint_positions = np.append(joint_positions, [0.0] * (nq_model - len(joint_positions)))
            elif len(joint_positions) > nq_model:
                joint_positions = joint_positions[:nq_model]

            gravity_torques = pin.computeGeneralizedGravity(self.robot.model, self.data, joint_positions)
            gravity_torques_6dof = gravity_torques[:6]

            compensated_torques = np.zeros(6)
            compensated_torques[0:3] = gravity_torques_6dof[0:3] / 4.0
            compensated_torques[3:6] = gravity_torques_6dof[3:6]

            output_msg = JointState()
            output_msg.header.stamp = rospy.Time.now()
            output_msg.header.frame_id = msg.header.frame_id if msg.header.frame_id else ""
            
            output_msg.name = list(msg.name[:7]) if len(msg.name) >= 7 else self.joint_names.copy()
            output_msg.position = list(msg.position[:7]) if len(msg.position) >= 7 else list(msg.position[:6]) + [0.0]
            output_msg.velocity = list(msg.velocity[:7]) if len(msg.velocity) >= 7 else (list(msg.velocity[:6]) + [0.0] if len(msg.velocity) >= 6 else [0.0] * 7)
            output_msg.effort = list(compensated_torques) + [0.0]

            self.torque_pub.publish(output_msg)

        except Exception as e:
            rospy.logerr("%s Error in callback: %s", self.name_id, str(e))


def check_ros_master():
    import rosnode
    try:
        rosnode.rosnode_ping("rosout", max_count=1, verbose=False)
    except rosnode.ROSNodeIOException:
        raise RuntimeError("ROS Master is not running.")


def main():
    try:
        check_ros_master()
        rospy.init_node("piper_gravity_compensation_node", anonymous=True)
        rospy.loginfo("Creating gravity compensation nodes for both robot arms...")
        robot_left = GravityCompensationArm(arm_side="left")
        robot_right = GravityCompensationArm(arm_side="right")
        rospy.loginfo("Gravity compensation nodes initialized successfully (zihao params: j1-3 /4, j4-6 x1)")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Node failed: %s", str(e))

if __name__ == "__main__":
    main()
