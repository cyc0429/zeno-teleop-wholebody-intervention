#!/usr/bin/env python3
"""
Evaluation and Visualization Script for MLP Reachability Map
=============================================================

This script helps verify the trained model and generate visualizations:

1. Sample the trained model on a regular 3D grid to visualize reachability
2. Compare model predictions against the original training data
3. Generate statistics on model accuracy and calibration
4. Save visualizations as PLY point clouds for inspection in Meshlab/RViz
5. Optionally display interactive 3D matplotlib visualizations in separate windows

Usage:
  python eval_reachability_mlp.py \
    --model_path ./trained_model/model.pth \
    --config_path ./trained_model/config.json \
    --hdf5_path /path/to/reachability.hdf5 \
    --output_dir ./eval_results \
    --grid_size 50 \
    --device cuda \
    --show_plots

Output:
  - grid_reachability.ply: 3D grid colored by predicted reachability
  - grid_manipulability.ply: 3D grid colored by predicted manipulability
  - training_data_predictions.ply: Training data colored by predictions vs ground truth
  - eval_stats.json: Quantitative evaluation metrics
  - eval_plots.png: Visualization of error distributions and calibration curves
  - Interactive matplotlib windows (if --show_plots is used):
    * Raw HDF5 data: 3D scatter plot colored by manipulability
    * Predicted grid: 3D scatter plot colored by reachability probability

Author: Teleop Team
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ReachabilityMLP(nn.Module):
    """MLP for reachability prediction (same as in training)."""
    
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=4, output_manip=True):
        super(ReachabilityMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_manip = output_manip
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        
        self.reach_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        if self.output_manip:
            self.manip_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()
            )
    
    def forward(self, xyz):
        x = self.backbone(xyz)
        p_reach = self.reach_head(x)
        m_pred = self.manip_head(x) if self.output_manip else None
        return p_reach, m_pred


def normalize_xyz(xyz, xyz_min, xyz_max):
    """Normalize xyz to [-1, 1]^3."""
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_half_size = (xyz_max - xyz_min) / 2
    return (xyz - xyz_center) / (xyz_half_size + 1e-8)


def save_ply(filename, points, labels=None, label_name='value'):
    """
    Save points to PLY file with optional labels as color.
    
    Args:
        filename: output PLY file path
        points: (N, 3) array of 3D points
        labels: (N,) array of scalar values to map to colors (0-1 range)
        label_name: description of the label for PLY header
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Map labels to RGB if provided
    if labels is not None:
        labels = np.clip(labels, 0, 1)  # Ensure [0, 1]
        # Use a simple colormap: red=low, green=mid, blue=high
        r = (1 - labels) * 255  # Red for low values
        g = (1 - np.abs(labels - 0.5) * 2) * 255  # Green for mid values
        b = labels * 255  # Blue for high values
        colors = np.column_stack([r, g, b]).astype(np.uint8)
    
    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment {label_name}\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if labels is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # Points
        if labels is not None:
            for i in range(len(points)):
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
        else:
            for i in range(len(points)):
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")
    
    print(f"Saved {len(points)} points to {filename}")


def visualize_raw_hdf5_data(xyz_train, manip_train, manip_max):
    """
    Visualize raw HDF5 training data as 3D scatter plot colored by manipulability.

    Args:
        xyz_train: (N, 3) array of training point coordinates
        manip_train: (N,) array of manipulability values
        manip_max: maximum manipulability value for normalization
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize manipulability to [0, 1] for coloring
    manip_norm = manip_train / (manip_max + 1e-8)

    # Create scatter plot
    scatter = ax.scatter(xyz_train[:, 0], xyz_train[:, 1], xyz_train[:, 2],
                        c=manip_norm, cmap='viridis', s=20, alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Manipulability Score (normalized)')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Raw HDF5 Training Data\n{len(xyz_train)} points')

    # Set equal aspect ratio
    max_range = np.array([xyz_train[:, 0].max() - xyz_train[:, 0].min(),
                         xyz_train[:, 1].max() - xyz_train[:, 1].min(),
                         xyz_train[:, 2].max() - xyz_train[:, 2].min()]).max() / 2.0

    mid_x = (xyz_train[:, 0].max() + xyz_train[:, 0].min()) * 0.5
    mid_y = (xyz_train[:, 1].max() + xyz_train[:, 1].min()) * 0.5
    mid_z = (xyz_train[:, 2].max() + xyz_train[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    return fig


def visualize_predicted_grid(grid_points_actual, p_reach_grid, grid_size):
    """
    Visualize predicted reachability on 3D grid as scatter plot colored by probability.

    Args:
        grid_points_actual: (N, 3) array of grid point coordinates
        p_reach_grid: (N,) array of predicted reachability probabilities
        grid_size: size of the grid per axis for title
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot
    scatter = ax.scatter(grid_points_actual[:, 0], grid_points_actual[:, 1], grid_points_actual[:, 2],
                        c=p_reach_grid, cmap='viridis', s=10, alpha=0.6)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Reachability Probability')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Predicted Reachability Grid\n{grid_size}³ = {len(grid_points_actual)} points')

    # Set equal aspect ratio
    max_range = np.array([grid_points_actual[:, 0].max() - grid_points_actual[:, 0].min(),
                         grid_points_actual[:, 1].max() - grid_points_actual[:, 1].min(),
                         grid_points_actual[:, 2].max() - grid_points_actual[:, 2].min()]).max() / 2.0

    mid_x = (grid_points_actual[:, 0].max() + grid_points_actual[:, 0].min()) * 0.5
    mid_y = (grid_points_actual[:, 1].max() + grid_points_actual[:, 1].min()) * 0.5
    mid_z = (grid_points_actual[:, 2].max() + grid_points_actual[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    return fig

def visualize_predicted_manipulability(grid_points_actual, m_pred_grid, grid_size):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot
    scatter = ax.scatter(grid_points_actual[:, 0], grid_points_actual[:, 1], grid_points_actual[:, 2],
                        c=m_pred_grid, cmap='viridis', s=10, alpha=0.6)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Manipulability Score')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Predicted Manipulability Grid\n{grid_size}³ = {len(grid_points_actual)} points')

    # Set equal aspect ratio
    max_range = np.array([grid_points_actual[:, 0].max() - grid_points_actual[:, 0].min(),
                         grid_points_actual[:, 1].max() - grid_points_actual[:, 1].min(),
                         grid_points_actual[:, 2].max() - grid_points_actual[:, 2].min()]).max() / 2.0

    mid_x = (grid_points_actual[:, 0].max() + grid_points_actual[:, 0].min()) * 0.5
    mid_y = (grid_points_actual[:, 1].max() + grid_points_actual[:, 1].min()) * 0.5
    mid_z = (grid_points_actual[:, 2].max() + grid_points_actual[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MLP reachability map')
    # parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    # parser.add_argument('--config_path', type=str, required=True, help='Path to config JSON')
    # parser.add_argument('--hdf5_path', type=str, required=True, help='Path to HDF5 reachability data')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--grid_size', type=int, default=30, help='Resolution of evaluation grid per axis')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for inference')
    # parser.add_argument('--show_plots', action='store_true', help='Show matplotlib 3D visualizations')
    
    args = parser.parse_args()
    args.model_path = "/home/jeong/zeno/wholebody-teleop/teleop/trained_model/model.pth"
    args.config_path = "/home/jeong/zeno/wholebody-teleop/teleop/trained_model/config.json"
    args.hdf5_path = "/home/jeong/zeno/wholebody-teleop/teleop/src/zeno-wholebody-teleop/robot_side/piper_reachable_region/maps/3D_reach_map_gripper_base_0.05_2025-12-11-17-29-20.h5"
    args.device = "cuda"
    args.show_plots = True
    
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading config...")
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    xyz_min = np.array(config['normalization']['xyz_min'])
    xyz_max = np.array(config['normalization']['xyz_max'])
    manip_max = config['normalization']['manip_max']
    
    print(f"Bounding box: {xyz_min} to {xyz_max}")
    
    # Load model
    print("Loading model...")
    model_config = config['model_architecture']
    model = ReachabilityMLP(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        output_manip=model_config['output_manip'],
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Generate evaluation grid
    print(f"Generating {args.grid_size}^3 evaluation grid...")
    grid_coords = np.linspace(0, 1, args.grid_size)
    grid_points = np.array(np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')).reshape(3, -1).T
    # Map from [0, 1]^3 to actual bounding box
    grid_points_actual = xyz_min + grid_points * (xyz_max - xyz_min)
    
    # Normalize for model input
    grid_points_norm = normalize_xyz(grid_points_actual, xyz_min, xyz_max)
    
    # Inference on grid
    print("Running inference on grid...")
    p_reach_grid = []
    m_pred_grid = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(grid_points_norm), args.batch_size)):
            batch = grid_points_norm[i:i+args.batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            p_reach_batch, m_pred_batch = model(batch_tensor)
            p_reach_grid.append(p_reach_batch.cpu().numpy())
            if m_pred_batch is not None:
                m_pred_grid.append(m_pred_batch.cpu().numpy())
    
    p_reach_grid = np.concatenate(p_reach_grid, axis=0).squeeze()
    m_pred_grid = np.concatenate(m_pred_grid, axis=0).squeeze() if m_pred_grid else None
    
    # Save grid visualizations
    print("Saving grid visualizations...")
    ply_reach_path = os.path.join(args.output_dir, 'grid_reachability.ply')
    save_ply(ply_reach_path, grid_points_actual, p_reach_grid, 'Reachability Probability')
    
    if m_pred_grid is not None:
        m_pred_grid_norm = m_pred_grid / (manip_max + 1e-8)  # Normalize to [0, 1] for visualization
        ply_manip_path = os.path.join(args.output_dir, 'grid_manipulability.ply')
        save_ply(ply_manip_path, grid_points_actual, m_pred_grid_norm, 'Manipulability Score')
    
    # Load training data and compare
    print("Loading training data for comparison...")
    with h5py.File(args.hdf5_path, 'r') as f:
        spheres = f['/Spheres/sphere_dataset'][()]
    
    xyz_train = spheres[:, :3]
    manip_train = spheres[:, 3]
    
    # Inference on training data
    print("Running inference on training data...")
    xyz_train_norm = normalize_xyz(xyz_train, xyz_min, xyz_max)
    
    with torch.no_grad():
        xyz_train_tensor = torch.from_numpy(xyz_train_norm).float().to(device)
        p_reach_train, m_pred_train = model(xyz_train_tensor)
        p_reach_train = p_reach_train.squeeze().cpu().numpy()
        m_pred_train = m_pred_train.squeeze().cpu().numpy() if m_pred_train is not None else None
    
    # Save training data visualization
    ply_train_path = os.path.join(args.output_dir, 'training_data_reachability.ply')
    save_ply(ply_train_path, xyz_train, p_reach_train, 'Model Predicted Reachability (training data)')
    
    # Compute statistics
    print("Computing statistics...")
    stats = {
        'grid_stats': {
            'num_points': len(grid_points_actual),
            'p_reach_mean': float(np.mean(p_reach_grid)),
            'p_reach_std': float(np.std(p_reach_grid)),
            'p_reach_min': float(np.min(p_reach_grid)),
            'p_reach_max': float(np.max(p_reach_grid)),
        },
        'training_data_stats': {
            'num_points': len(xyz_train),
            'p_reach_mean': float(np.mean(p_reach_train)),
            'p_reach_std': float(np.std(p_reach_train)),
            'p_reach_min': float(np.min(p_reach_train)),
            'p_reach_max': float(np.max(p_reach_train)),
            'percent_high_confidence': float(100 * np.sum(p_reach_train > 0.9) / len(p_reach_train)),
        },
    }
    
    if m_pred_train is not None:
        manip_train_norm = manip_train / (manip_max + 1e-8)
        stats['manipulability_stats'] = {
            'prediction_min': float(np.min(m_pred_train)),
            'prediction_max': float(np.max(m_pred_train)),
            'prediction_mean': float(np.mean(m_pred_train)),
            'prediction_std': float(np.std(m_pred_train)),
            'label_mean': float(np.mean(manip_train_norm)),
            'label_std': float(np.std(manip_train_norm)),
            'mae': float(np.mean(np.abs(m_pred_train - manip_train_norm))),
            'rmse': float(np.sqrt(np.mean((m_pred_train - manip_train_norm) ** 2))),
        }
    
    stats_path = os.path.join(args.output_dir, 'eval_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Create matplotlib visualizations if requested
    if args.show_plots:
        print("Creating matplotlib visualizations...")

        # Visualize raw HDF5 data
        fig_raw = visualize_raw_hdf5_data(xyz_train, manip_train, manip_max)

        # Visualize predicted grid
        fig_pred = visualize_predicted_grid(grid_points_actual, p_reach_grid, args.grid_size)

        # Visualize predicted manipulability
        fig_pred_manip = visualize_predicted_manipulability(grid_points_actual, m_pred_grid, args.grid_size)

        # Show plots
        plt.show()

    print("\nStatistics:")
    print(json.dumps(stats, indent=2))
    print(f"\nSaved statistics to {stats_path}")
    print("Evaluation complete!")


if __name__ == '__main__':
    main()

