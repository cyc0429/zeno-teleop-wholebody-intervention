#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gravity compensation torque calculation node for dual-arm robot
Subscribes to joint angles from both arms, computes and publishes gravity compensation torques
Gripper is included in calculation but output contains only first 6 joints' torques
Joint1-3: divided by 4, Joint4-6: unchanged
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
        """
        Initialize gravity compensation node

        Args:
            arm_side: Arm side, 'left' or 'right'
        """
        if arm_side not in ["left", "right"]:
            raise ValueError("arm_side must be 'left' or 'right'")

        self.arm_side = arm_side

        # Get URDF file path and resolve package:// paths
        try:
            package_path = subprocess.check_output("rospack find piper_description", shell=True).strip().decode("utf-8")
            urdf_path = os.path.join(package_path, "urdf", "piper_description.urdf")
            urdf_path = os.path.abspath(urdf_path)
            rospy.loginfo("[%s] URDF path: %s", self.arm_side, urdf_path)

            # Read URDF file content
            with open(urdf_path, "r") as f:
                urdf_content = f.read()

            # Replace package:// paths with absolute paths
            def resolve_package_path(match):
                package_name = match.group(1)
                relative_path = match.group(2)
                try:
                    pkg_path = (
                        subprocess.check_output(["rospack", "find", package_name], stderr=subprocess.DEVNULL)
                        .strip()
                        .decode("utf-8")
                    )
                    absolute_path = os.path.join(pkg_path, relative_path)
                    return absolute_path
                except Exception as e:
                    rospy.logwarn(
                        "[%s] Failed to resolve package://%s/%s: %s", self.arm_side, package_name, relative_path, str(e)
                    )
                    return match.group(0)  # Return original if resolution fails

            # Replace all package:// paths in URDF content
            urdf_content = re.sub(r"package://([^/]+)/(.+)", resolve_package_path, urdf_content)

            # Write processed URDF to temporary file
            temp_urdf_file = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
            temp_urdf_file.write(urdf_content)
            temp_urdf_file.close()
            processed_urdf_path = temp_urdf_file.name
            rospy.loginfo("[%s] Processed URDF saved to: %s", self.arm_side, processed_urdf_path)

        except Exception as e:
            rospy.logerr("[%s] Failed to process URDF file: %s", self.arm_side, str(e))
            raise

        # Load robot model
        try:
            self.robot = pin.RobotWrapper.BuildFromURDF(processed_urdf_path)
            rospy.loginfo("[%s] Robot model loaded successfully", self.arm_side)
            rospy.loginfo("[%s] Number of joints: %d", self.arm_side, self.robot.model.nq)

            # Clean up temporary file
            try:
                os.unlink(processed_urdf_path)
            except Exception:
                pass
        except Exception as e:
            rospy.logerr("[%s] Failed to load robot model: %s", self.arm_side, str(e))
            # Clean up temporary file on error
            try:
                os.unlink(processed_urdf_path)
            except Exception:
                pass
            raise

        # Use full model (including gripper) for calculation
        # Gripper gravity affects first 6 joints' torques, but output contains only first 6 joints
        rospy.loginfo("[%s] Using full robot model (including gripper) for gravity compensation", self.arm_side)

        # Create data object
        self.data = self.robot.model.createData()

        # Subscribe to joint states
        self.joint_state_sub = rospy.Subscriber(
            f"/robot/arm_{arm_side}/joint_states_single", JointState, self.joint_state_callback, queue_size=1
        )

        # Publish gravity compensation torques
        self.torque_pub = rospy.Publisher(f"/robot/arm_{arm_side}/joint_states_compensated", JointState, queue_size=1)

        # Joint names (6 joints + gripper)
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

        rospy.loginfo("[%s] Gravity compensation node initialized", self.arm_side)
        rospy.loginfo("[%s] Subscribing to: /robot/arm_%s/joint_states_single", self.arm_side, arm_side)
        rospy.loginfo("[%s] Publishing to: /robot/arm_%s/joint_states_compensated", self.arm_side, arm_side)

    def joint_state_callback(self, msg):
        """Joint state callback: compute and publish gravity compensation torques"""
        try:
            if len(msg.position) < 6:
                rospy.logwarn("[%s] Received joint state with less than 6 joints", self.arm_side)
                return

            # Get all joint angles (including gripper)
            if len(msg.position) >= 7:
                joint_positions = np.array(msg.position[:7])
            else:
                joint_positions = np.array(list(msg.position[:6]) + [0.0])

            # Pad or truncate to match model joint count
            nq_model = self.robot.model.nq
            if len(joint_positions) < nq_model:
                joint_positions = np.append(joint_positions, [0.0] * (nq_model - len(joint_positions)))
            elif len(joint_positions) > nq_model:
                joint_positions = joint_positions[:nq_model]

            # Compute gravity compensation torques
            gravity_torques = pin.computeGeneralizedGravity(self.robot.model, self.data, joint_positions)

            # Extract first 6 joints' torques
            gravity_torques_6dof = gravity_torques[:6]

            # Apply scaling: joint1-3 divided by 4, joint4-6 unchanged
            compensated_torques = np.zeros(6)
            compensated_torques[0:3] = gravity_torques_6dof[0:3] / 4.0
            compensated_torques[3:6] = gravity_torques_6dof[3:6]

            # Create output message
            output_msg = JointState()
            output_msg.header.stamp = rospy.Time.now()
            output_msg.header.frame_id = msg.header.frame_id if msg.header.frame_id else ""

            # Set joint names
            if len(msg.name) >= 7:
                output_msg.name = list(msg.name[:7])
            else:
                output_msg.name = self.joint_names.copy()

            # Set positions
            if len(msg.position) >= 7:
                output_msg.position = list(msg.position[:7])
            else:
                output_msg.position = list(msg.position[:6]) + [0.0]

            # Set velocities
            if len(msg.velocity) >= 7:
                output_msg.velocity = list(msg.velocity[:7])
            elif len(msg.velocity) >= 6:
                output_msg.velocity = list(msg.velocity[:6]) + [0.0]
            else:
                output_msg.velocity = [0.0] * 7

            # Set torques (first 6 joints compensated, gripper = 0)
            output_msg.effort = list(compensated_torques) + [0.0]

            # Publish message
            self.torque_pub.publish(output_msg)

        except Exception as e:
            rospy.logerr("[%s] Error in joint_state_callback: %s", self.arm_side, str(e))


def check_ros_master():
    """Check if ROS master is running"""
    import rosnode

    try:
        rosnode.rosnode_ping("rosout", max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")


def main():
    """Main function: create gravity compensation nodes for both arms"""
    try:
        check_ros_master()

        rospy.init_node("piper_gravity_compensation_node", anonymous=True)

        rospy.loginfo("Creating gravity compensation nodes for both arms...")
        arm_left = GravityCompensationArm(arm_side="left")
        arm_right = GravityCompensationArm(arm_side="right")

        rospy.loginfo("Gravity compensation nodes initialized successfully")
        rospy.loginfo("Gripper gravity included in calculation")
        rospy.loginfo("Joint1-3: divided by 4, Joint4-6: unchanged")

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Node failed: %s", str(e))


if __name__ == "__main__":
    main()
