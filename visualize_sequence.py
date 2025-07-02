#!/usr/bin/env python3

import os
import glob
import open3d as o3d
import time
import argparse


def visualize_pointcloud_sequence(pointcloud_dir, fps=5, auto_play=True):
    """
    Visualize a sequence of point clouds with frame-by-frame navigation.
    
    Args:
        pointcloud_dir: Directory containing PLY files
        fps: Frames per second for auto-play mode
        auto_play: Whether to automatically cycle through frames
    """
    # Get all PLY files
    ply_files = sorted(glob.glob(os.path.join(pointcloud_dir, "*.ply")))
    
    if not ply_files:
        print(f"No PLY files found in {pointcloud_dir}")
        return
    
    print(f"Found {len(ply_files)} point cloud files")
    print("Controls:")
    print("  - Space: Pause/Resume auto-play")
    print("  - Right Arrow / N: Next frame")
    print("  - Left Arrow / P: Previous frame")
    print("  - Q / Escape: Quit")
    print("  - R: Reset view")
    
    # Initialize visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Sequence Viewer", width=1200, height=800)
    
    # Load first point cloud
    current_idx = 0
    pcd = o3d.io.read_point_cloud(ply_files[current_idx])
    vis.add_geometry(pcd)
    
    # State variables
    is_playing = auto_play
    last_update_time = time.time()
    frame_interval = 1.0 / fps
    
    def update_pointcloud(idx):
        """Update the displayed point cloud"""
        nonlocal current_idx
        if 0 <= idx < len(ply_files):
            current_idx = idx
            new_pcd = o3d.io.read_point_cloud(ply_files[current_idx])
            
            if len(new_pcd.points) > 0:
                # Update geometry
                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors
                vis.update_geometry(pcd)
                
                # Update window title
                filename = os.path.basename(ply_files[current_idx])
                vis.get_render_option().window_title = f"Frame {current_idx + 1}/{len(ply_files)}: {filename}"
            
            print(f"Frame {current_idx + 1}/{len(ply_files)}: {os.path.basename(ply_files[current_idx])}")
    
    # Key callbacks
    def toggle_play(vis):
        nonlocal is_playing
        is_playing = not is_playing
        print("Auto-play:", "ON" if is_playing else "OFF")
        return False
    
    def next_frame(vis):
        update_pointcloud((current_idx + 1) % len(ply_files))
        return False
    
    def prev_frame(vis):
        update_pointcloud((current_idx - 1) % len(ply_files))
        return False
    
    def quit_viewer(vis):
        vis.destroy_window()
        return True
    
    def reset_view(vis):
        vis.reset_view_point(True)
        return False
    
    # Register key callbacks
    vis.register_key_callback(ord(" "), toggle_play)  # Space
    vis.register_key_callback(262, next_frame)        # Right arrow
    vis.register_key_callback(263, prev_frame)        # Left arrow  
    vis.register_key_callback(ord("N"), next_frame)   # N
    vis.register_key_callback(ord("P"), prev_frame)   # P
    vis.register_key_callback(ord("Q"), quit_viewer)  # Q
    vis.register_key_callback(256, quit_viewer)       # Escape
    vis.register_key_callback(ord("R"), reset_view)   # R
    
    # Set initial view
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_up([0, -1, 0])
    
    # Main visualization loop
    while True:
        # Auto-play logic
        if is_playing:
            current_time = time.time()
            if current_time - last_update_time >= frame_interval:
                update_pointcloud((current_idx + 1) % len(ply_files))
                last_update_time = current_time
        
        # Update visualization
        if not vis.poll_events():
            break
        vis.update_renderer()
    
    vis.destroy_window()


def create_animation_video(pointcloud_dir, output_video="pointcloud_sequence.mp4", fps=10):
    """
    Create a video from the point cloud sequence (requires ffmpeg).
    
    Args:
        pointcloud_dir: Directory containing PLY files
        output_video: Output video filename
        fps: Video framerate
    """
    import tempfile
    import subprocess
    
    ply_files = sorted(glob.glob(os.path.join(pointcloud_dir, "*.ply")))
    
    if not ply_files:
        print(f"No PLY files found in {pointcloud_dir}")
        return
    
    print(f"Creating video from {len(ply_files)} point clouds...")
    
    # Create temporary directory for images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Render each point cloud to image
        for i, ply_file in enumerate(ply_files):
            pcd = o3d.io.read_point_cloud(ply_file)
            
            if len(pcd.points) == 0:
                continue
            
            # Create visualizer for rendering
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1920, height=1080)
            vis.add_geometry(pcd)
            
            # Set view
            vis.get_view_control().set_front([0, 0, -1])
            vis.get_view_control().set_up([0, -1, 0])
            vis.get_view_control().set_zoom(0.7)
            
            # Capture image
            image_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            vis.capture_screen_image(image_path, do_render=True)
            vis.destroy_window()
            
            if i % 10 == 0:
                print(f"Rendered {i + 1}/{len(ply_files)} frames")
        
        # Create video with ffmpeg
        try:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                output_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Video saved as: {output_video}")
            else:
                print(f"FFmpeg error: {result.stderr}")
                
        except FileNotFoundError:
            print("FFmpeg not found. Please install ffmpeg to create videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize point cloud sequences")
    parser.add_argument("pointcloud_dir", help="Directory containing PLY files")
    parser.add_argument("--fps", type=float, default=5, help="Frames per second for auto-play")
    parser.add_argument("--no-auto-play", action="store_true", help="Disable auto-play")
    parser.add_argument("--create-video", action="store_true", help="Create MP4 video instead of interactive viewer")
    parser.add_argument("--video-fps", type=float, default=10, help="Video framerate")
    parser.add_argument("--output-video", default="pointcloud_sequence.mp4", help="Output video filename")
    
    args = parser.parse_args()
    
    if args.create_video:
        create_animation_video(args.pointcloud_dir, args.output_video, args.video_fps)
    else:
        visualize_pointcloud_sequence(args.pointcloud_dir, args.fps, not args.no_auto_play)