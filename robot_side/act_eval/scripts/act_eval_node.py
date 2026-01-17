#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT Evaluation ROS Node

Subscribes to:
- /robot/arm_left/joint_states_single
- /robot/arm_right/joint_states_single
- /realsense_left/color/image_raw/compressed
- /realsense_right/color/image_raw/compressed
- /realsense_top/color/image_raw/compressed

Publishes:
- /robot/arm_left/act_joint_cmd
- /robot/arm_right/act_joint_cmd
"""

import sys
import os
import rospy
import numpy as np
import cv2
import torch
import pickle
from threading import Lock
from sensor_msgs.msg import JointState, CompressedImage
from cv_bridge import CvBridge
from einops import rearrange
import torchvision.transforms as transforms

# Add act directory to path to import modules
act_dir = os.path.join(os.path.dirname(__file__), '../../../../../../act')
sys.path.insert(0, act_dir)

from constants import DT
from policy import ACTPolicy, CNNMLPPolicy
from utils import pad_qpos, set_seed


class ACTEvalNode:
    def __init__(self):
        rospy.init_node('act_eval_node', anonymous=True)
        
        # Get ROS parameters (same as imitate_episodes.py)
        self.ckpt_dir = rospy.get_param('~ckpt_dir')
        self.task_name = rospy.get_param('~task_name')
        self.policy_class = rospy.get_param('~policy_class', 'ACT')
        self.use_qtor = rospy.get_param('~use_qtor', False)
        self.use_lidar = rospy.get_param('~use_lidar', False)
        self.mix = rospy.get_param('~mix', False)
        self.temporal_agg = rospy.get_param('~temporal_agg', False)
        self.seed = rospy.get_param('~seed', 1000)
        
        # ACT-specific parameters
        self.chunk_size = rospy.get_param('~chunk_size', 100)
        self.kl_weight = rospy.get_param('~kl_weight', 10)
        self.hidden_dim = rospy.get_param('~hidden_dim', 512)
        self.dim_feedforward = rospy.get_param('~dim_feedforward', 3200)
        
        # Get task configuration
        is_sim = self.task_name[:4] == 'sim_'
        if is_sim:
            from constants import SIM_TASK_CONFIGS
            task_config = SIM_TASK_CONFIGS[self.task_name]
        else:
            from constants import REAL_TASK_CONFIGS
            task_config = REAL_TASK_CONFIGS[self.task_name]
        
        self.camera_names = task_config['camera_names']
        self.state_dim = rospy.get_param('~state_dim', 17)
        
        # Fixed parameters
        lr_backbone = 1e-5
        backbone = 'resnet18'
        if self.policy_class == 'ACT':
            enc_layers = 4
            dec_layers = 7
            nheads = 8
            policy_config = {
                'lr': 1e-5,  # Not used during inference
                'num_queries': self.chunk_size,
                'kl_weight': self.kl_weight,
                'hidden_dim': self.hidden_dim,
                'dim_feedforward': self.dim_feedforward,
                'lr_backbone': lr_backbone,
                'backbone': backbone,
                'enc_layers': enc_layers,
                'dec_layers': dec_layers,
                'nheads': nheads,
                'camera_names': self.camera_names,
                'state_dim': self.state_dim,
                'use_lidar': self.use_lidar,
            }
        elif self.policy_class == 'CNNMLP':
            policy_config = {
                'lr': 1e-5,
                'lr_backbone': lr_backbone,
                'backbone': backbone,
                'num_queries': 1,
                'camera_names': self.camera_names,
            }
        else:
            raise NotImplementedError(f"Policy class {self.policy_class} not supported")
        
        # Load policy and stats
        set_seed(self.seed)
        self.policy = self._make_policy(self.policy_class, policy_config)
        ckpt_path = os.path.join(self.ckpt_dir, 'policy_best.ckpt')
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        rospy.loginfo(f"Model loading status: {loading_status}")
        self.policy.cuda()
        self.policy.eval()
        rospy.loginfo(f'Loaded model from: {ckpt_path}')
        
        # Load dataset stats
        stats_path = os.path.join(self.ckpt_dir, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        # Pad stats if needed (for backward compatibility)
        self._pad_stats()
        
        # Image preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.bridge = CvBridge()
        
        # Data storage with locks
        self.data_lock = Lock()
        self.joint_states_left = None
        self.joint_states_right = None
        self.images = {cam: None for cam in self.camera_names}
        self.qtor_left = None
        self.qtor_right = None
        self.lidar_scan = None
        
        # Temporal aggregation setup
        self.query_frequency = self.chunk_size if not self.temporal_agg else 1
        if self.temporal_agg:
            self.num_queries = self.chunk_size
            self.all_time_actions = torch.zeros([10000, 10000 + self.num_queries, self.state_dim]).cuda()
        self.timestep = 0
        self.all_actions = None
        
        # Publishers
        self.pub_left = rospy.Publisher('/robot/arm_left/act_joint_cmd', JointState, queue_size=1)
        self.pub_right = rospy.Publisher('/robot/arm_right/act_joint_cmd', JointState, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/robot/arm_left/joint_states_single', JointState, 
                         self.joint_state_left_callback, queue_size=1)
        rospy.Subscriber('/robot/arm_right/joint_states_single', JointState,
                         self.joint_state_right_callback, queue_size=1)
        
        # Map camera names to topics
        camera_topic_map = {
            'left': '/realsense_left/color/image_raw/compressed',
            'right': '/realsense_right/color/image_raw/compressed',
            'top': '/realsense_top/color/image_raw/compressed'
        }
        
        for cam_name in self.camera_names:
            if cam_name in camera_topic_map:
                rospy.Subscriber(camera_topic_map[cam_name], CompressedImage,
                               lambda msg, name=cam_name: self.image_callback(msg, name), queue_size=1)
        
        rospy.loginfo("ACT Eval Node initialized")
        rospy.loginfo(f"Task: {self.task_name}, Policy: {self.policy_class}")
        rospy.loginfo(f"Cameras: {self.camera_names}")
        
        # Control loop timer
        self.control_rate = rospy.Rate(1.0 / 10)  # Match simulation DT
        self.control_timer = rospy.Timer(self.control_rate, self.control_loop)
    
    def _make_policy(self, policy_class, policy_config):
        if policy_class == 'ACT':
            return ACTPolicy(policy_config)
        elif policy_class == 'CNNMLP':
            return CNNMLPPolicy(policy_config)
        else:
            raise NotImplementedError
    
    def _pad_stats(self):
        """Pad stats if they are 14D (for backward compatibility)"""
        if self.stats['qpos_mean'].shape[-1] < 17:
            pad_size = 17 - self.stats['qpos_mean'].shape[-1]
            qpos_mean_padding = np.zeros(pad_size)
            qpos_std_padding = np.ones(pad_size)
            self.stats['qpos_mean'] = np.concatenate([qpos_mean_padding, self.stats['qpos_mean']])
            self.stats['qpos_std'] = np.concatenate([qpos_std_padding, self.stats['qpos_std']])
            if 'example_qpos' in self.stats and self.stats['example_qpos'].shape[-1] < 17:
                example_qpos_padding = np.zeros((*self.stats['example_qpos'].shape[:-1], pad_size))
                self.stats['example_qpos'] = np.concatenate([example_qpos_padding, self.stats['example_qpos']], axis=-1)
        
        if self.stats['action_mean'].shape[-1] < 17:
            pad_size = 17 - self.stats['action_mean'].shape[-1]
            action_mean_padding = np.zeros(pad_size)
            action_std_padding = np.ones(pad_size)
            self.stats['action_mean'] = np.concatenate([action_mean_padding, self.stats['action_mean']])
            self.stats['action_std'] = np.concatenate([action_std_padding, self.stats['action_std']])
    
    def _pre_process(self, qpos):
        """Preprocess qpos: pad if needed, handle mix parameter, then normalize."""
        qpos_padded = pad_qpos(qpos, target_dim=17)
        if self.state_dim == 17:
            if not self.mix:
                # Set first 3 values to zeros (ignore base position)
                if isinstance(qpos_padded, torch.Tensor):
                    qpos_padded = qpos_padded.clone()
                    qpos_padded[..., :3] = 0.0
                else:
                    qpos_padded = qpos_padded.copy()
                    qpos_padded[..., :3] = 0.0
        return (qpos_padded - self.stats['qpos_mean']) / self.stats['qpos_std']
    
    def _post_process(self, action):
        """Post-process action: pad if needed, denormalize, then remove padding if needed."""
        if action.shape[-1] < 17:
            action = pad_qpos(action, target_dim=17)
        action_denorm = action * self.stats['action_std'] + self.stats['action_mean']
        if action_denorm.shape[-1] == 17 and self.state_dim != 17:
            action_denorm = action_denorm[..., 3:]  # Remove first 3 padded zeros
        return action_denorm
    
    def joint_state_left_callback(self, msg):
        with self.data_lock:
            self.joint_states_left = msg
            if self.use_qtor and len(msg.effort) >= len(msg.position):
                self.qtor_left = np.array(msg.effort[:len(msg.position)], dtype=np.float32)
    
    def joint_state_right_callback(self, msg):
        with self.data_lock:
            self.joint_states_right = msg
            if self.use_qtor and len(msg.effort) >= len(msg.position):
                self.qtor_right = np.array(msg.effort[:len(msg.position)], dtype=np.float32)
    
    def image_callback(self, msg, cam_name):
        """Decode compressed image and store."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with self.data_lock:
                    self.images[cam_name] = img_rgb
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Error decoding image for {cam_name}: {e}")
    
    def _get_image_tensor(self):
        """Get current images as tensor."""
        curr_images = []
        with self.data_lock:
            for cam_name in self.camera_names:
                if self.images[cam_name] is not None:
                    img = self.images[cam_name]
                    # Resize if needed (assuming model expects specific size)
                    # For now, use image as-is
                    curr_image = rearrange(img, 'h w c -> c h w')
                    curr_images.append(curr_image)
                else:
                    # Return None if images not ready
                    return None
        
        if len(curr_images) != len(self.camera_names):
            return None
        
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        curr_image = self.normalize(curr_image)
        return curr_image
    
    def control_loop(self, event):
        """Main control loop - runs at DT frequency."""
        with self.data_lock:
            # Check if we have all required data
            if self.joint_states_left is None or self.joint_states_right is None:
                return
            
            # Check if all images are available
            if any(self.images[cam] is None for cam in self.camera_names):
                return
        
        # Get images as tensor
        curr_image = self._get_image_tensor()
        if curr_image is None:
            return
        
        # Process joint states
        with self.data_lock:
            qpos_left_np = np.array(self.joint_states_left.position, dtype=np.float32)
            qpos_right_np = np.array(self.joint_states_right.position, dtype=np.float32)
            
            # Combine left and right qpos first
            # For dual arm: typically 7 per arm (6 joints + 1 gripper) = 14 total
            qpos_combined = np.concatenate([qpos_left_np, qpos_right_np])
            
            # Handle mix parameter for state_dim==17 (mobile robots)
            # mix parameter replaces first 3 values (base position) with qvel[:3] (base velocity)
            # Note: For mobile robots, base velocity should come from a single source (not per arm)
            # For now, we'll use left arm's velocity for base if available, or zeros
            if self.state_dim == 17 and self.mix:
                qvel_left_np = np.array(self.joint_states_left.velocity, dtype=np.float32)
                # Pad qpos_combined to 17D first (add 3 zeros for base position)
                if len(qpos_combined) < 17:
                    qpos_combined = pad_qpos(qpos_combined, target_dim=17)
                # Replace first 3 values (base position) with base velocity
                if len(qvel_left_np) >= 3:
                    qpos_combined[:3] = qvel_left_np[:3]
                else:
                    # If no velocity available, set to zeros (will be handled in pre_process)
                    qpos_combined[:3] = 0.0
            
            # Pad to state_dim if needed (for non-mobile or when mix=False)
            if len(qpos_combined) < self.state_dim:
                qpos_combined = pad_qpos(qpos_combined, target_dim=self.state_dim)
            elif len(qpos_combined) > self.state_dim:
                # Truncate if too long (shouldn't happen, but safety check)
                qpos_combined = qpos_combined[:self.state_dim]
        
        # Preprocess qpos
        qpos = self._pre_process(qpos_combined)
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        
        # Get qtor if needed
        if self.use_qtor:
            with self.data_lock:
                if self.qtor_left is not None and self.qtor_right is not None:
                    qtor_combined = np.concatenate([self.qtor_left, self.qtor_right])
                    if len(qtor_combined) < self.state_dim:
                        qtor_combined = pad_qpos(qtor_combined, target_dim=self.state_dim)
                    elif len(qtor_combined) > self.state_dim:
                        qtor_combined = qtor_combined[:self.state_dim]
                    qtor = self._pre_process(qtor_combined)
                    qtor = torch.from_numpy(qtor).float().cuda().unsqueeze(0)
                else:
                    qtor = torch.zeros_like(qpos)
        else:
            qtor = torch.zeros_like(qpos)
        
        # Get lidar scan if needed
        if self.use_lidar:
            with self.data_lock:
                if self.lidar_scan is not None:
                    lidar_scan = torch.from_numpy(self.lidar_scan).float().cuda().unsqueeze(0)
                else:
                    lidar_scan = torch.zeros((1, 1080), dtype=torch.float32).cuda()
        else:
            lidar_scan = torch.zeros((1, 1080), dtype=torch.float32).cuda()
        
        # Query policy
        with torch.inference_mode():
            if self.policy_class == "ACT":
                if self.temporal_agg:
                    if self.timestep % self.query_frequency == 0:
                        all_actions = self.policy(qpos, curr_image, qtor=qtor, lidar_scan=lidar_scan)
                        self.all_time_actions[[self.timestep], self.timestep:self.timestep+self.num_queries] = all_actions
                    
                    actions_for_curr_step = self.all_time_actions[:, self.timestep]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    
                    if len(actions_for_curr_step) > 0:
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        return  # Not enough actions yet
                    
                    self.timestep += 1
                else:
                    if self.timestep % self.query_frequency == 0:
                        self.all_actions = self.policy(qpos, curr_image, qtor=qtor, lidar_scan=lidar_scan)
                    raw_action = self.all_actions[:, self.timestep % self.query_frequency]
                    self.timestep += 1
            elif self.policy_class == "CNNMLP":
                raw_action = self.policy(qpos, curr_image, qtor=qtor, lidar_scan=lidar_scan)
            else:
                raise NotImplementedError
        
        # Post-process action
        raw_action = raw_action.squeeze(0).cpu().numpy()
        action = self._post_process(raw_action)
        
        # Split action into left and right
        # For dual arm: typically 14D (7 per arm) or 17D (3 base + 14 arm)
        # Remove base velocities if present (first 3 values for mobile)
        action_arm_only = action
        if len(action) == 17 and self.state_dim == 17:
            # Remove first 3 base velocities, keep 14 arm actions
            action_arm_only = action[3:]
        
        # Split into left and right (assuming equal split)
        if len(action_arm_only) >= 14:
            # Dual arm: 7 joints per arm
            mid_point = len(action_arm_only) // 2
            action_left = action_arm_only[:mid_point]
            action_right = action_arm_only[mid_point:]
        elif len(action_arm_only) == 7:
            # Single arm mode: apply same action to both
            action_left = action_arm_only
            action_right = action_arm_only
        else:
            rospy.logwarn_throttle(5.0, f"Unexpected action dimension: {len(action)} (arm_only: {len(action_arm_only)})")
            return
        
        # Publish commands
        msg_left = JointState()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.position = action_left.tolist()
        
        msg_right = JointState()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.position = action_right.tolist()
        
        self.pub_left.publish(msg_left)
        self.pub_right.publish(msg_right)


def main():
    try:
        node = ACTEvalNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

