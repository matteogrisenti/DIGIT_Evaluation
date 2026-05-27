"""
compile_raw_video.py
-------
Utility script to compile a directory of individual raw frames (PNG/JPG) 
into a single MP4 video file.
"""

import os
import argparse
import glob
import cv2

def compile_video(input_dir: str, output_video: str | None, fps: float) -> None:
    """Read sequenced images from a directory and write them to an MP4 video."""
    
    # 1. Validate Input
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # 2. Gather and Sort Frames
    # We look for common image formats. The zero-padded filenames (frame_000000.png) 
    # ensure that standard alphabetical sorting works perfectly.
    search_pattern = os.path.join(input_dir, "frame_*.png")
    frame_files = sorted(glob.glob(search_pattern))
    
    if not frame_files:
        # Fallback to jpg if pngs aren't found
        search_pattern = os.path.join(input_dir, "frame_*.jpg")
        frame_files = sorted(glob.glob(search_pattern))

    num_frames = len(frame_files)
    if num_frames == 0:
        print(f"Error: No frames found in '{input_dir}' matching 'frame_*.png' or 'frame_*.jpg'.")
        return
        
    print(f"Found {num_frames} frames in '{input_dir}'.")

    # 3. Dynamic Output Path Setup
    if not output_video:
        # Get the parent directory of the frames folder
        parent_dir = os.path.dirname(os.path.abspath(input_dir))
        # Get the name of the folder itself (e.g., "raw_frames_20260527_103000")
        folder_name = os.path.basename(os.path.normpath(input_dir))
        # Create output path: parent_dir/raw_frames_20260527_103000_compiled.mp4
        output_video = os.path.join(parent_dir, f"{folder_name}_compiled.mp4")

    # 4. Read First Frame to get Dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Error: Could not read the first frame ({frame_files[0]}). Is it corrupted?")
        return
        
    height, width, _ = first_frame.shape

    # 5. Setup Video Writer
    print(f"Initializing VideoWriter: {width}x{height} resolution at {fps} FPS.")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 6. Process Frames
    print("Compiling video... this may take a moment depending on the number of frames.")
    
    try:
        for i, frame_path in enumerate(frame_files):
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"Warning: Skipping unreadable frame '{frame_path}'.")
                continue
                
            video_writer.write(frame)
            
            # Print progress every 100 frames to keep the terminal clean
            if (i + 1) % 100 == 0 or (i + 1) == num_frames:
                print(f"Processed {i + 1}/{num_frames} frames...")

    except Exception as e:
        print(f"\nError during compilation: {e}")
        
    finally:
        # 7. Clean up
        video_writer.release()
        print(f"\nSuccess! Video saved to: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile a directory of image frames into an MP4 video.")
    
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, 
        help="Path to the directory containing the frames (e.g., experiments_output/2026.../raw_frames_...)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default=None, 
        help="Optional: Path to save the output MP4 video. If omitted, saves next to the input directory."
    )
    
    parser.add_argument(
        "--fps", 
        type=float, 
        default=30.0, 
        help="Frames per second for the output video (default: 30.0)"
    )
    
    args = parser.parse_args()
    
    compile_video(args.input, args.output, args.fps)