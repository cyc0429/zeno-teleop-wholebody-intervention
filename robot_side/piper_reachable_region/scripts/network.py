#!/usr/bin/env python3
"""
Shared Network Definition for Reachability MLP
==============================================

This module contains the shared ReachabilityMLP network architecture used across
training, inference, and ROS nodes.

Author: Teleop Team
"""

import torch
import torch.nn as nn


class ReachabilityMLP(nn.Module):
    """
    MLP-based implicit reachability function with dual heads.
    
    Architecture:
      - Shared backbone: several fully-connected layers with ReLU activation
      - Head 1 (Reachability): outputs sigmoid-normalized probability [0, 1]
      - Head 2 (Manipulability): outputs non-negative manipulability score
    
    This network is used for:
      - Training: train_reachability_mlp.py
      - Inference: reachability_mask_node.py, manipulability_base_control_node.py
    """
    
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=4, output_manip=True):
        """
        Args:
            input_dim: dimensionality of input (typically 3 for xyz)
            hidden_dim: hidden layer size
            num_layers: number of layers in the backbone
            output_manip: whether to output manipulability head
        """
        super(ReachabilityMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_manip = output_manip
        
        # Build shared backbone
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.backbone = nn.Sequential(*layers)
        
        # Reachability head: outputs probability [0, 1]
        self.reach_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Manipulability head: outputs non-negative score
        if self.output_manip:
            self.manip_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()  # Softplus ensures output >= 0
            )
    
    def forward(self, xyz):
        """
        Forward pass through the network.
        
        Args:
            xyz: (B, 3) tensor of normalized 3D positions
            
        Returns:
            p_reach: (B, 1) tensor of reachability probabilities
            m_pred: (B, 1) tensor of manipulability scores (or None if output_manip=False)
        """
        x = self.backbone(xyz)
        p_reach = self.reach_head(x)
        m_pred = self.manip_head(x) if self.output_manip else None
        return p_reach, m_pred

