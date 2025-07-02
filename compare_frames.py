#!/usr/bin/env python3

import os
import glob
import open3d as o3d
import argparse
import numpy as np


def compare_pointcloud_frames(pointcloud_dir, frame_indices=None, grid_size=2, headless=False, output_image=None):
    """
    Compare multiple point cloud frames side by side.
    
    Args:
        pointcloud_dir: Directory containing PLY files
        frame_indices: List of frame indices to compare (None for first few frames)
        grid_size: Number of frames to show in grid layout
        headless: Whether to run in headless mode (save image instead of display)
        output_image: Output image filename for headless mode
    """
    ply_files = sorted(glob.glob(os.path.join(pointcloud_dir, "*.ply")))
    
    if not ply_files:
        print(f"No PLY files found in {pointcloud_dir}")
        return
    
    if frame_indices is None:
        # Show first, middle, and last frames by default
        n = len(ply_files)
        frame_indices = [0, n//4, n//2, 3*n//4, n-1][:grid_size*grid_size]
    
    # Limit to available frames
    frame_indices = [i for i in frame_indices if 0 <= i < len(ply_files)]
    
    print(f"Comparing frames: {frame_indices}")
    
    # Create visualizer
    try:
        if headless:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1600, height=1200)
        else:
            vis = o3d.visualization.Visualizer()
            success = vis.create_window(window_name="Point Cloud Frame Comparison", width=1600, height=1200)
            if not success:
                print("Failed to create window. Switching to headless mode...")
                return compare_pointcloud_frames(pointcloud_dir, frame_indices, grid_size, True, output_image)
    except Exception as e:
        print(f"Failed to create visualizer: {e}")
        print("Switching to headless mode...")
        return compare_pointcloud_frames(pointcloud_dir, frame_indices, grid_size, True, output_image)
    
    # Load and position point clouds
    offset_x = 3.0  # Spacing between point clouds
    
    for i, frame_idx in enumerate(frame_indices):
        pcd = o3d.io.read_point_cloud(ply_files[frame_idx])
        
        if len(pcd.points) == 0:
            continue
        
        # Translate point cloud to avoid overlap
        translation = [i * offset_x, 0, 0]
        pcd.translate(translation)
        
        # Add frame number as text (simulate with colored points)
        vis.add_geometry(pcd)
        
        print(f"Frame {frame_idx}: {os.path.basename(ply_files[frame_idx])}")
    
    # Set view
    try:
        view_control = vis.get_view_control()
        if view_control is not None:
            view_control.set_front([0, 0, -1])
            view_control.set_up([0, -1, 0])
            view_control.set_zoom(0.5)
    except Exception as e:
        print(f"Warning: Could not set view: {e}")
    
    if headless:
        # Render and save image
        if output_image is None:
            output_image = "pointcloud_comparison.png"
        
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_image, do_render=True)
        print(f"Saved comparison image to: {output_image}")
        vis.destroy_window()
    else:
        # Run interactive visualizer
        vis.run()
        vis.destroy_window()


def analyze_pointcloud_stats(pointcloud_dir, frame_indices=None):
    """
    Analyze and compare statistics of point clouds instead of visual comparison.
    
    Args:
        pointcloud_dir: Directory containing PLY files
        frame_indices: List of frame indices to analyze
    """
    ply_files = sorted(glob.glob(os.path.join(pointcloud_dir, "*.ply")))
    
    if not ply_files:
        print(f"No PLY files found in {pointcloud_dir}")
        return
    
    if frame_indices is None:
        # Analyze first few frames by default
        n = len(ply_files)
        frame_indices = [0, n//4, n//2, 3*n//4, n-1][:5]
    
    frame_indices = [i for i in frame_indices if 0 <= i < len(ply_files)]
    
    print(f"\nPoint Cloud Statistics Comparison")
    print("=" * 80)
    print(f"{'Frame':<8} {'Filename':<25} {'Points':<10} {'Bounds (X,Y,Z)':<30} {'Center':<20}")
    print("-" * 80)
    
    for frame_idx in frame_indices:
        pcd = o3d.io.read_point_cloud(ply_files[frame_idx])
        filename = os.path.basename(ply_files[frame_idx])
        
        if len(pcd.points) == 0:
            print(f"{frame_idx:<8} {filename:<25} {'0':<10} {'Empty':<30} {'N/A':<20}")
            continue
        
        points = np.asarray(pcd.points)
        num_points = len(points)
        
        # Calculate bounds
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        bounds_str = f"[{min_bounds[0]:.1f},{max_bounds[0]:.1f}] [{min_bounds[1]:.1f},{max_bounds[1]:.1f}] [{min_bounds[2]:.1f},{max_bounds[2]:.1f}]"
        
        # Calculate center
        center = points.mean(axis=0)
        center_str = f"({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})"
        
        print(f"{frame_idx:<8} {filename:<25} {num_points:<10} {bounds_str:<30} {center_str:<20}")
    
    print("-" * 80)
    
    # Calculate movement between frames
    if len(frame_indices) > 1:
        print(f"\nMovement Analysis:")
        print("=" * 50)
        
        prev_pcd = None
        prev_idx = None
        
        for frame_idx in frame_indices:
            pcd = o3d.io.read_point_cloud(ply_files[frame_idx])
            
            if len(pcd.points) == 0:
                continue
                
            if prev_pcd is not None and len(prev_pcd.points) > 0:
                prev_points = np.asarray(prev_pcd.points)
                curr_points = np.asarray(pcd.points)
                
                prev_center = prev_points.mean(axis=0)
                curr_center = curr_points.mean(axis=0)
                
                displacement = np.linalg.norm(curr_center - prev_center)
                direction = curr_center - prev_center
                
                print(f"Frame {prev_idx} -> {frame_idx}: Displacement = {displacement:.3f} units")
                print(f"  Direction: ({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})")
            
            prev_pcd = pcd
            prev_idx = frame_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare point cloud frames")
    parser.add_argument("pointcloud_dir", help="Directory containing PLY files")
    parser.add_argument("--frames", nargs="+", type=int, help="Frame indices to compare")
    parser.add_argument("--grid-size", type=int, default=2, help="Grid size for layout")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (save image)")
    parser.add_argument("--output-image", help="Output image filename for headless mode")
    parser.add_argument("--stats-only", action="store_true", help="Only show statistics, no visualization")
    
    args = parser.parse_args()
    
    if args.stats_only:
        analyze_pointcloud_stats(args.pointcloud_dir, args.frames)
    else:
        compare_pointcloud_frames(args.pointcloud_dir, args.frames, args.grid_size, args.headless, args.output_image)