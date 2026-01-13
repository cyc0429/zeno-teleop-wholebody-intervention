#!/usr/bin/env python3
"""
ROS Node for Manipulability-Based Base Control
===============================================

This ROS node:
  1. Subscribes to /robot/arm_left/joint_states_single and /robot/arm_right/joint_states_single
  2. Computes manipulability in real time using trained MLP model
  3. If manipulability is smaller than a threshold, computes the composed Intent direction (both arms)
     and maps it into base velocity command

Configuration via ROS parameters:
  ~model_path: Path to trained PyTorch model (model.pth)
  ~config_path: Path to model config (config.json)
  ~urdf_path: Path to robot URDF file
  ~manip_threshold: Manipulability threshold below which base control is activated
  ~base_vel_topic: Topic for publishing base velocity commands (default: /cmd_vel)
  ~max_linear_vel: Maximum linear velocity (m/s)
  ~max_angular_vel: Maximum angular velocity (rad/s)
  ~device: PyTorch device ('cpu' or 'cuda')

Usage:
  rosrun piper_reachable_region manipulability_base_control_node.py

Author: Teleop Team
"""

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import rospy
import torch
import pytorch_kinematics as pk

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

# Import shared network definition
from network import ReachabilityMLP


class ManipulabilityBaseControlNode:
    """
    ROS node for manipulability-based base control.
    """
    
    def __init__(self):
        """
        Initialize the node: load model, set up kinematics, create subscribers/publishers.
        """
        rospy.loginfo("Initializing ManipulabilityBaseControlNode...")
        
        # Get parameters
        self.model_path = rospy.get_param('~model_path', 'model.pth')
        self.config_path = rospy.get_param('~config_path', 'config.json')
        self.urdf_path = rospy.get_param('~urdf_path', '')
        self.manip_threshold = rospy.get_param('~manip_threshold', 0.1)
        self.base_vel_topic = rospy.get_param('~base_vel_topic', '/robot/intent_vel')
        self.max_linear_vel = rospy.get_param('~max_linear_vel', 0.3)
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 0.5)
        self.stretch_radius = rospy.get_param('~stretch_radius', 0.3)  # Minimum distance from base to consider arm stretched
        device_str = rospy.get_param('~device', 'cpu')
        
        # End-effector link names
        self.ee_link_left = rospy.get_param('~ee_link_left', 'gripper_base')
        self.ee_link_right = rospy.get_param('~ee_link_right', 'gripper_base')
        self.base_link = rospy.get_param('~base_link', 'base_link')
        
        rospy.loginfo(f"Model path: {self.model_path}")
        rospy.loginfo(f"Config path: {self.config_path}")
        rospy.loginfo(f"URDF path: {self.urdf_path}")
        rospy.loginfo(f"Manipulability threshold: {self.manip_threshold}")
        rospy.loginfo(f"Base velocity topic: {self.base_vel_topic}")
        rospy.loginfo(f"Max linear velocity: {self.max_linear_vel} m/s")
        rospy.loginfo(f"Max angular velocity: {self.max_angular_vel} rad/s")
        rospy.loginfo(f"Stretch radius threshold: {self.stretch_radius} m")
        
        # Setup device
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
        rospy.loginfo(f"Using device: {self.device}")
        
        # Load config
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            rospy.loginfo("Config loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load config: {e}")
            raise
        
        # Extract normalization parameters
        self.xyz_min = np.array(self.config['normalization']['xyz_min'])
        self.xyz_max = np.array(self.config['normalization']['xyz_max'])
        self.manip_max = self.config['normalization']['manip_max']
        self.manip_min = self.config['normalization']['manip_min']
        
        # Load model
        try:
            model_config = self.config['model_architecture']
            self.model = ReachabilityMLP(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                output_manip=model_config['output_manip'],
            ).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            rospy.loginfo("Model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            raise
        
        # Load URDF and build kinematic chains
        if not self.urdf_path or not os.path.exists(self.urdf_path):
            # Try to find default URDF
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            default_urdf = os.path.abspath(
                os.path.join(repo_root, "..", "piper_ros", "src", "piper_description", "urdf", "piper_description.urdf")
            )
            if os.path.exists(default_urdf):
                self.urdf_path = default_urdf
            else:
                rospy.logerr(f"URDF file not found. Please set ~urdf_path parameter.")
                raise FileNotFoundError("URDF file not found")
        
        rospy.loginfo(f"Loading URDF from: {self.urdf_path}")
        try:
            with open(self.urdf_path, 'rb') as f:
                urdf_bytes = f.read()
            
            # Build kinematic chains for both arms
            # For dual-arm setups, both arms typically use the same end-effector link name
            # but may have different base links or be in different namespaces
            # We'll try to build chains with the specified end-effector names
            try:
                self.chain_left = pk.build_serial_chain_from_urdf(urdf_bytes, self.ee_link_left)
                self.chain_left = self.chain_left.to(dtype=torch.float32, device=self.device)
                rospy.loginfo(f"Left arm chain built with end-effector: {self.ee_link_left}")
            except Exception as e:
                rospy.logwarn(f"Failed to build left arm chain with {self.ee_link_left}: {e}")
                # Try alternative end-effector names
                for alt_name in ['link6', 'link7', 'gripper_base']:
                    try:
                        self.chain_left = pk.build_serial_chain_from_urdf(urdf_bytes, alt_name)
                        self.chain_left = self.chain_left.to(dtype=torch.float32, device=self.device)
                        rospy.loginfo(f"Left arm chain built with alternative end-effector: {alt_name}")
                        break
                    except:
                        continue
                else:
                    raise Exception("Could not build left arm kinematic chain")
            
            # Build right arm chain (may use same or different end-effector name)
            try:
                if self.ee_link_right != self.ee_link_left:
                    self.chain_right = pk.build_serial_chain_from_urdf(urdf_bytes, self.ee_link_right)
                    self.chain_right = self.chain_right.to(dtype=torch.float32, device=self.device)
                    rospy.loginfo(f"Right arm chain built with end-effector: {self.ee_link_right}")
                else:
                    # Same end-effector name, reuse chain structure
                    self.chain_right = self.chain_left
                    rospy.loginfo("Right arm chain reusing left arm structure")
            except Exception as e:
                rospy.logwarn(f"Failed to build right arm chain with {self.ee_link_right}: {e}")
                # Use left chain as fallback (assuming symmetric arms)
                rospy.logwarn("Using left arm chain structure for right arm")
                self.chain_right = self.chain_left
            
            rospy.loginfo("Kinematic chains built successfully")
        except Exception as e:
            rospy.logerr(f"Failed to build kinematic chains: {e}")
            traceback.print_exc()
            raise
        
        # Joint state storage
        self.left_joint_positions = None
        self.right_joint_positions = None
        self.left_joint_states_received = False
        self.right_joint_states_received = False
        
        # Setup subscribers and publishers
        self.left_joint_sub = rospy.Subscriber(
            '/robot/arm_left/joint_states_single', 
            JointState, 
            self.left_joint_callback, 
            queue_size=1
        )
        self.right_joint_sub = rospy.Subscriber(
            '/robot/arm_right/joint_states_single', 
            JointState, 
            self.right_joint_callback, 
            queue_size=1
        )
        self.cmd_vel_pub = rospy.Publisher(self.base_vel_topic, Twist, queue_size=1)
        
        rospy.loginfo("ManipulabilityBaseControlNode initialized successfully")
    
    def normalize_xyz(self, xyz):
        """
        Normalize 3D positions to [-1, 1]^3 using stored bounding box.
        
        Args:
            xyz: (N, 3) array of 3D positions in base frame
            
        Returns:
            xyz_normalized: (N, 3) array normalized to [-1, 1]^3
        """
        xyz_center = (self.xyz_min + self.xyz_max) / 2
        xyz_half_size = (self.xyz_max - self.xyz_min) / 2
        xyz_norm = (xyz - xyz_center) / (xyz_half_size + 1e-8)
        return np.clip(xyz_norm, -1.0, 1.0)  # Clip to [-1, 1]
    
    def denormalize_manip(self, m_norm):
        """
        Denormalize manipulability from normalized space to original scale.
        
        Args:
            m_norm: (N,) normalized manipulability scores
            
        Returns:
            m: (N,) denormalized manipulability
        """
        manip_range = self.manip_max - self.manip_min
        m = self.manip_min + m_norm * manip_range
        return m
    
    def left_joint_callback(self, msg):
        """Callback for left arm joint states."""
        if len(msg.position) >= 6:
            self.left_joint_positions = np.array(msg.position[:6], dtype=np.float32)
            self.left_joint_states_received = True
            self.compute_and_publish_base_vel()
    
    def right_joint_callback(self, msg):
        """Callback for right arm joint states."""
        if len(msg.position) >= 6:
            self.right_joint_positions = np.array(msg.position[:6], dtype=np.float32)
            self.right_joint_states_received = True
            self.compute_and_publish_base_vel()
    
    def compute_end_effector_position(self, chain, joint_positions):
        """
        Compute end-effector position from joint positions using forward kinematics.
        
        Args:
            chain: pytorch_kinematics chain
            joint_positions: (6,) array of joint positions
            
        Returns:
            ee_pos: (3,) array of end-effector position [x, y, z] in base frame
        """
        try:
            joint_tensor = torch.from_numpy(joint_positions).float().unsqueeze(0).to(self.device)
            transform = chain.forward_kinematics(joint_tensor)
            ee_pos = transform.get_matrix()[0, :3, 3].cpu().numpy()
            return ee_pos
        except Exception as e:
            rospy.logwarn(f"Failed to compute FK: {e}")
            return None
    
    def compute_manipulability_from_model(self, ee_pos):
        """
        Compute manipulability at end-effector position using trained MLP model.
        
        Args:
            ee_pos: (3,) array of end-effector position [x, y, z]
            
        Returns:
            manipulability: scalar manipulability score
        """
        try:
            # Normalize position
            ee_pos_norm = self.normalize_xyz(ee_pos.reshape(1, -1))
            
            # Query model
            with torch.no_grad():
                ee_pos_tensor = torch.from_numpy(ee_pos_norm).float().to(self.device)
                p_reach, m_pred = self.model(ee_pos_tensor)
                
                if m_pred is not None:
                    m_pred = m_pred.squeeze().cpu().numpy()
                    # Denormalize manipulability
                    m_pred = self.denormalize_manip(m_pred)
                    return float(m_pred)
                else:
                    return 0.0
        except Exception as e:
            rospy.logwarn(f"Failed to compute manipulability from model: {e}")
            return 0.0
    
    def is_arm_stretched(self, ee_pos):
        """
        Check if an arm is stretched out based on distance from base.
        
        An arm is considered "stretched out" if its end-effector is beyond
        a certain radius from the base (origin).
        
        Args:
            ee_pos: (3,) array of end-effector position [x, y, z] in base frame
            
        Returns:
            bool: True if arm is stretched out, False otherwise
        """
        # Compute distance from base (origin) to end-effector
        distance = np.linalg.norm(ee_pos)
        return distance > self.stretch_radius
    
    def compute_intent_direction(self, ee_pos_left, ee_pos_right, manip_left, manip_right, 
                                 left_stretched, right_stretched):
        """
        Compute composed intent direction from end-effector positions using weighted sum.
        Only considers arms that are stretched out.
        
        The intent direction is computed as a weighted sum of the end-effector positions,
        where weights are inversely proportional to manipulability (lower manipulability
        gets higher weight, prioritizing arms that need more help).
        The direction is projected onto the xy plane and normalized.
        
        Args:
            ee_pos_left: (3,) array of left end-effector position
            ee_pos_right: (3,) array of right end-effector position
            manip_left: manipulability of left arm
            manip_right: manipulability of right arm
            left_stretched: bool, whether left arm is stretched out
            right_stretched: bool, whether right arm is stretched out
            
        Returns:
            intent_dir: (2,) array of intent direction [x, y] in base frame (normalized)
                       Returns [0, 0] if no arms are stretched out
        """
        # Only consider stretched-out arms
        if not left_stretched and not right_stretched:
            # No arms stretched out, return zero direction
            return np.array([0.0, 0.0])
        
        # Compute manipulability deficits (how far below threshold)
        # Use a small epsilon to avoid division by zero
        epsilon = 1e-6
        manip_deficit_left = max(epsilon, self.manip_threshold - manip_left) if left_stretched else 0.0
        manip_deficit_right = max(epsilon, self.manip_threshold - manip_right) if right_stretched else 0.0
        
        # Compute weights: higher deficit = higher weight
        # Weights are proportional to manipulability deficit
        weight_left = manip_deficit_left
        weight_right = manip_deficit_right
        
        # Normalize weights so they sum to 1
        total_weight = weight_left + weight_right
        if total_weight > epsilon:
            weight_left = weight_left / total_weight
            weight_right = weight_right / total_weight
        else:
            # Fallback to equal weights if both are at threshold
            if left_stretched and right_stretched:
                weight_left = 0.5
                weight_right = 0.5
            elif left_stretched:
                weight_left = 1.0
                weight_right = 0.0
            else:  # right_stretched
                weight_left = 0.0
                weight_right = 1.0
        
        # Compute weighted average end-effector position
        ee_pos_weighted = weight_left * ee_pos_left + weight_right * ee_pos_right
        
        # Project to xy plane (ignore z)
        intent_dir_xy = ee_pos_weighted[:2]
        
        # Normalize direction
        norm = np.linalg.norm(intent_dir_xy)
        if norm > 1e-6:
            intent_dir_xy = intent_dir_xy / norm
        else:
            intent_dir_xy = np.array([0.0, 0.0])
        
        return intent_dir_xy
    
    def map_intent_to_base_vel(self, intent_dir, manip_left, manip_right):
        """
        Map intent direction to base velocity command.
        
        Args:
            intent_dir: (2,) array of intent direction [x, y]
            manip_left: manipulability of left arm
            manip_right: manipulability of right arm
            
        Returns:
            cmd_vel: Twist message with base velocity command
        """
        cmd_vel = Twist()
        
        # Compute average manipulability
        avg_manip = (manip_left + manip_right) / 2.0
        
        # Scale velocity based on how far below threshold we are
        # More aggressive control when manipulability is very low
        # manip_deficit = max(0.0, self.manip_threshold - avg_manip)
        # scale_factor = min(1.0, manip_deficit / (self.manip_threshold - 0.01))
        scale_factor = 1.0
        
        # Map intent direction to base velocities
        # x direction: forward/backward
        # y direction: left/right (lateral)
        # For mobile base: typically cmd_vel.linear.x is forward, cmd_vel.linear.y is lateral
        
        # Scale by max velocity and deficit
        linear_vel_x = intent_dir[0] * self.max_linear_vel * scale_factor
        linear_vel_y = intent_dir[1] * self.max_linear_vel * scale_factor
        
        # Clamp to max velocities
        linear_vel_x = np.clip(linear_vel_x, -self.max_linear_vel, self.max_linear_vel)
        linear_vel_y = np.clip(linear_vel_y, -self.max_linear_vel, self.max_linear_vel)
        
        cmd_vel.linear.x = float(linear_vel_x)
        cmd_vel.linear.y = float(linear_vel_y)
        cmd_vel.linear.z = 0.0
        cmd_vel.angular.x = 0.0
        cmd_vel.angular.y = 0.0
        cmd_vel.angular.z = 0.0  # No rotation for now, can be added if needed
        
        return cmd_vel
    
    def compute_and_publish_base_vel(self):
        """
        Main computation function: compute manipulability and publish base velocity if needed.
        """
        # Check if we have both joint states
        if not (self.left_joint_states_received and self.right_joint_states_received):
            return
        
        if self.left_joint_positions is None or self.right_joint_positions is None:
            return
        
        try:
            # Compute end-effector positions
            ee_pos_left = self.compute_end_effector_position(self.chain_left, self.left_joint_positions)
            ee_pos_right = self.compute_end_effector_position(self.chain_right, self.right_joint_positions)
            
            if ee_pos_left is None or ee_pos_right is None:
                return
            
            # Check if arms are stretched out (sphere check)
            left_stretched = self.is_arm_stretched(ee_pos_left)
            right_stretched = self.is_arm_stretched(ee_pos_right)
            
            rospy.logdebug(f"Arm stretch status - Left: {left_stretched} (dist: {np.linalg.norm(ee_pos_left):.3f}), Right: {right_stretched} (dist: {np.linalg.norm(ee_pos_right):.3f})")
            
            # Only proceed if at least one arm is stretched out
            if not (left_stretched or right_stretched):
                # No arms stretched out, publish zero velocity
                cmd_vel = Twist()
                self.cmd_vel_pub.publish(cmd_vel)
                rospy.logdebug("No arms stretched out, publishing zero velocity")
                return
            
            # Compute manipulability from model (only for stretched-out arms)
            manip_left = self.compute_manipulability_from_model(ee_pos_left) if left_stretched else self.manip_threshold
            manip_right = self.compute_manipulability_from_model(ee_pos_right) if right_stretched else self.manip_threshold
            
            rospy.loginfo(f"Manipulability - Left: {manip_left:.4f} (stretched: {left_stretched}), Right: {manip_right:.4f} (stretched: {right_stretched})")
            
            # Compute average manipulability only for stretched-out arms
            if left_stretched and right_stretched:
                avg_manip = (manip_left + manip_right) / 2.0
            elif left_stretched:
                avg_manip = manip_left
            else:  # right_stretched
                avg_manip = manip_right
            
            # Check if manipulability is below threshold
            if avg_manip < self.manip_threshold:
                # Compute intent direction (weighted by manipulability, only considering stretched arms)
                intent_dir = self.compute_intent_direction(
                    ee_pos_left, ee_pos_right, manip_left, manip_right,
                    left_stretched, right_stretched
                )
                
                # Only publish if intent direction is non-zero
                if np.linalg.norm(intent_dir) > 1e-6:
                    # Map to base velocity command
                    cmd_vel = self.map_intent_to_base_vel(intent_dir, manip_left, manip_right)
                    
                    # Publish base velocity command
                    self.cmd_vel_pub.publish(cmd_vel)
                    
                    rospy.logdebug(
                        f"Low manipulability ({avg_manip:.4f} < {self.manip_threshold}). "
                        f"Publishing base vel: linear.x={cmd_vel.linear.x:.3f}, linear.y={cmd_vel.linear.y:.3f}"
                    )
                else:
                    # Zero intent direction, publish zero velocity
                    cmd_vel = Twist()
                    self.cmd_vel_pub.publish(cmd_vel)
            else:
                # Publish zero velocity when manipulability is sufficient
                cmd_vel = Twist()
                self.cmd_vel_pub.publish(cmd_vel)
                
        except Exception as e:
            rospy.logerr(f"Error in compute_and_publish_base_vel: {e}")
            traceback.print_exc()


def main():
    """
    Main entry point for the ROS node.
    """
    rospy.init_node('manipulability_base_control_node')
    
    try:
        node = ManipulabilityBaseControlNode()
        rospy.loginfo("Spinning...")
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()

