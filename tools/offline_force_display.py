"""
offline_force_display.py
-------
Entry point for the offline display of force field data.

It loads a .npz dataset containing the normal and shear forces, and visualizes them using 
the same display tools as the live application, saving the playback to an MP4 file 
directly inside the experiment folder.
"""

import os
import argparse
import sys
import numpy as np
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Adjust the import path if your utils module is structured differently
from utils.display_tools import build_combined_view, WINDOW_MULTI, teardown_windows


def play_and_record(dataset_path: str, output_video: str | None, target_fps: float) -> None:
    """Load the .npz dataset, render it to the screen, and write to a video file."""
    
    # 1. Load Data
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found.")
        return

    # 2. Dynamic Output Path Setup
    # If no output path is provided, save it in the same directory as the input dataset
    if not output_video:
        experiment_dir = os.path.dirname(os.path.abspath(dataset_path))
        dataset_basename = os.path.splitext(os.path.basename(dataset_path))[0]
        # Example: if input is "force_record_123.npz", output becomes "force_record_123_playback.mp4"
        output_video = os.path.join(experiment_dir, f"{dataset_basename}_playback.mp4")

    print(f"Loading dataset from: {dataset_path}")
    data = np.load(dataset_path)
    
    normal_frames = data['normal']
    shear_frames = data['shear']
    timestamps = data['timestamps']
    
    num_frames = len(normal_frames)
    print(f"Loaded {num_frames} frames successfully.")

    if num_frames == 0:
        print("Dataset is empty. Exiting.")
        return

    # 3. Setup the dummy raw frame
    # We create a dummy raw frame to feed into build_combined_view.
    # Standard DIGIT resolution is 320x240 (or 240x320 depending on rotation).
    dummy_raw_frame = np.zeros((320, 240, 3), dtype=np.uint8)

    # 4. Setup Video Writer
    # Do a single dry-run render on the first frame to extract the exact output dimensions
    test_combined = build_combined_view(normal_frames[0], shear_frames[0], dummy_raw_frame)
    out_h, out_w, _ = test_combined.shape

    print(f"Initializing VideoWriter: {out_w}x{out_h} resolution at {target_fps} FPS.")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, target_fps, (out_w, out_h))

    # Calculate delay based on target FPS (defaulting to 1ms if FPS is too high)
    frame_delay_ms = max(1, int(1000 / target_fps))

    print("\nStarting playback. Press ESC in the display window to quit early.")

    # 5. Rendering Loop
    try:
        for i in range(num_frames):
            norm = normal_frames[i]
            shear = shear_frames[i]
            
            # Build the visualization just like the live pipeline
            combined = build_combined_view(norm, shear, dummy_raw_frame)
            
            # Save frame to output video
            video_writer.write(combined)

            # Display to screen
            cv2.imshow(WINDOW_MULTI, combined)

            # Wait and check for quit signal (ESC key)
            if cv2.waitKey(frame_delay_ms) == 27: 
                print("\nPlayback interrupted by user.")
                break
                
    except Exception as e:
        print(f"\nError during playback: {e}")
        
    finally:
        # 6. Clean up resources
        video_writer.release()
        teardown_windows()
        print(f"Video successfully saved to: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline force display and video generation for force field .npz datasets.")
    
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, 
        help="Path to the .npz dataset file (e.g., experiments_output/20260527_090000/force_record_...).npz"
    )
    
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default=None, 
        help="Optional: Path to save the output MP4 video. If omitted, saves directly in the input dataset's folder."
    )
    
    parser.add_argument(
        "--fps", 
        type=float, 
        default=30.0, 
        help="Playback and output video frames per second (default: 30.0)"
    )
    
    args = parser.parse_args()
    
    play_and_record(args.input, args.output, args.fps)