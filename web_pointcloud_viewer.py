#!/usr/bin/env python3

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import viser
import viser.transforms as viser_tf
import open3d as o3d
from tqdm.auto import tqdm


def web_pointcloud_sequence_viewer(
    pointcloud_dir: str,
    port: int = 8080,
    auto_play: bool = True,
    fps: float = 5.0,
    point_size: float = 0.002,
):
    """
    Web-based point cloud sequence viewer using viser.
    
    Args:
        pointcloud_dir: Directory containing PLY files
        port: Port number for the viser server
        auto_play: Whether to start in auto-play mode
        fps: Frames per second for auto-play
        point_size: Size of points in the visualization
    """
    print(f"Starting web point cloud viewer on port {port}")
    
    # Get all PLY files
    ply_files = sorted(glob.glob(os.path.join(pointcloud_dir, "*.ply")))
    
    if not ply_files:
        print(f"No PLY files found in {pointcloud_dir}")
        return
    
    print(f"Found {len(ply_files)} point cloud files")
    
    # Load all point clouds
    print("Loading point clouds...")
    point_clouds = []
    valid_indices = []
    
    for i, ply_file in enumerate(tqdm(ply_files)):
        try:
            pcd = o3d.io.read_point_cloud(ply_file)
            if len(pcd.points) > 0:
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                
                # If no colors, use default
                if len(colors) == 0:
                    colors = np.ones_like(points) * 0.7
                else:
                    # Ensure colors are in [0,1] range
                    if colors.max() > 1.0:
                        colors = colors / 255.0
                
                point_clouds.append({
                    'points': points,
                    'colors': colors,
                    'filename': os.path.basename(ply_file)
                })
                valid_indices.append(i)
            else:
                print(f"Warning: Empty point cloud in {ply_file}")
        except Exception as e:
            print(f"Error loading {ply_file}: {e}")
    
    if not point_clouds:
        print("No valid point clouds found!")
        return
    
    print(f"Loaded {len(point_clouds)} valid point clouds")
    
    # Calculate scene bounds for centering
    all_points = np.concatenate([pc['points'] for pc in point_clouds], axis=0)
    scene_center = np.mean(all_points, axis=0)
    scene_bounds = np.max(np.abs(all_points - scene_center))
    
    # Center all point clouds
    for pc in point_clouds:
        pc['points'] = pc['points'] - scene_center
    
    # Start viser server
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    
    # State variables
    current_frame = 0
    is_playing = auto_play
    last_update_time = time.time()
    frame_interval = 1.0 / fps
    
    # GUI Controls
    with server.gui.add_folder("Playback Controls"):
        play_button = server.gui.add_button("⏸️ Pause" if is_playing else "▶️ Play")
        prev_button = server.gui.add_button("⏮️ Previous")
        next_button = server.gui.add_button("⏩ Next")
        
        frame_slider = server.gui.add_slider(
            "Frame", 
            min=0, 
            max=len(point_clouds) - 1, 
            step=1, 
            initial_value=0
        )
        
        fps_slider = server.gui.add_slider(
            "FPS", 
            min=0.1, 
            max=30.0, 
            step=0.1, 
            initial_value=fps
        )
    
    with server.gui.add_folder("Display Settings"):
        point_size_slider = server.gui.add_slider(
            "Point Size", 
            min=0.0001, 
            max=0.01, 
            step=0.0001, 
            initial_value=point_size
        )
        
        show_info = server.gui.add_checkbox("Show Frame Info", initial_value=True)
    
    # Frame info display
    frame_info = server.gui.add_text("Frame Info", initial_value="")
    
    # Create initial point cloud
    point_cloud_handle = server.scene.add_point_cloud(
        name="sequence_pcd",
        points=point_clouds[0]['points'],
        colors=point_clouds[0]['colors'],
        point_size=point_size,
        point_shape="circle",
    )
    
    def update_frame_info():
        """Update the frame information display"""
        if show_info.value:
            pc = point_clouds[current_frame]
            info_text = (
                f"Frame: {current_frame + 1}/{len(point_clouds)}\n"
                f"File: {pc['filename']}\n"
                f"Points: {len(pc['points']):,}\n"
                f"Playing: {'Yes' if is_playing else 'No'}\n"
                f"FPS: {fps_slider.value:.1f}"
            )
            frame_info.value = info_text
        else:
            frame_info.value = ""
    
    def update_point_cloud(frame_idx: int):
        """Update the displayed point cloud"""
        nonlocal current_frame
        
        if 0 <= frame_idx < len(point_clouds):
            current_frame = frame_idx
            pc = point_clouds[current_frame]
            
            # Update point cloud
            point_cloud_handle.points = pc['points']
            point_cloud_handle.colors = pc['colors']
            point_cloud_handle.point_size = point_size_slider.value
            
            # Update frame info
            update_frame_info()
            
            print(f"Displaying frame {current_frame + 1}/{len(point_clouds)}: {pc['filename']}")
    
    def update_point_cloud_from_autoplay(frame_idx: int):
        """Update point cloud from auto-play without triggering slider callback"""
        nonlocal current_frame
        
        if 0 <= frame_idx < len(point_clouds):
            current_frame = frame_idx
            pc = point_clouds[current_frame]
            
            # Update point cloud
            point_cloud_handle.points = pc['points']
            point_cloud_handle.colors = pc['colors']
            point_cloud_handle.point_size = point_size_slider.value
            
            # Don't update slider to avoid recursion
            
            # Update frame info
            update_frame_info()
            
            print(f"Displaying frame {current_frame + 1}/{len(point_clouds)}: {pc['filename']}")
    
    # Button callbacks
    @play_button.on_click
    def _(_):
        nonlocal is_playing
        is_playing = not is_playing
        play_button.text = "⏸️ Pause" if is_playing else "▶️ Play"
        update_frame_info()
    
    @prev_button.on_click
    def _(_):
        new_frame = (current_frame - 1) % len(point_clouds)
        update_point_cloud(new_frame)
    
    @next_button.on_click
    def _(_):
        new_frame = (current_frame + 1) % len(point_clouds)
        update_point_cloud(new_frame)
    
    # Slider callbacks
    @frame_slider.on_update
    def _(_):
        update_point_cloud(int(frame_slider.value))
    
    @fps_slider.on_update
    def _(_):
        nonlocal frame_interval
        frame_interval = 1.0 / fps_slider.value
        update_frame_info()
    
    @point_size_slider.on_update
    def _(_):
        point_cloud_handle.point_size = point_size_slider.value
    
    @show_info.on_update
    def _(_):
        update_frame_info()
    
    # Initial setup
    update_frame_info()
    
    # Auto-play loop
    def auto_play_loop():
        nonlocal last_update_time, current_frame
        
        while True:
            if is_playing and len(point_clouds) > 1:
                current_time = time.time()
                if current_time - last_update_time >= frame_interval:
                    new_frame = (current_frame + 1) % len(point_clouds)
                    update_point_cloud_from_autoplay(new_frame)
                    last_update_time = current_time
            
            time.sleep(0.01)  # Small sleep to prevent excessive CPU usage
    
    # Start auto-play thread
    auto_play_thread = threading.Thread(target=auto_play_loop, daemon=True)
    auto_play_thread.start()
    
    print(f"\n🌐 Web viewer started!")
    print(f"📍 Open your browser and go to: http://localhost:{port}")
    print(f"📁 Viewing {len(point_clouds)} point clouds from: {pointcloud_dir}")
    print("\n🎮 Controls:")
    print("  - Use the web interface buttons and sliders to navigate")
    print("  - Play/Pause: Toggle auto-play")
    print("  - Previous/Next: Navigate frames manually")
    print("  - Frame slider: Jump to specific frame")
    print("  - FPS slider: Adjust playback speed")
    print("  - Point Size slider: Adjust point size")
    print("\n💡 Tip: You can interact with the 3D view using mouse:")
    print("  - Left click + drag: Rotate")
    print("  - Right click + drag: Pan")
    print("  - Scroll: Zoom")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    try:
        # Keep server running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        server.stop()


def main():
    """Main function for the web point cloud sequence viewer."""
    parser = argparse.ArgumentParser(description="Web-based point cloud sequence viewer")
    parser.add_argument("pointcloud_dir", help="Directory containing PLY files")
    parser.add_argument("--port", type=int, default=8080, help="Port number for the web server")
    parser.add_argument("--fps", type=float, default=5.0, help="Initial frames per second for auto-play")
    parser.add_argument("--no-auto-play", action="store_true", help="Start with auto-play disabled")
    parser.add_argument("--point-size", type=float, default=0.002, help="Initial point size")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.pointcloud_dir):
        print(f"Error: Directory '{args.pointcloud_dir}' does not exist")
        return
    
    web_pointcloud_sequence_viewer(
        pointcloud_dir=args.pointcloud_dir,
        port=args.port,
        auto_play=not args.no_auto_play,
        fps=args.fps,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()