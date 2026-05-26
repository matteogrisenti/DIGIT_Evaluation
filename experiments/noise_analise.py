import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from utils.draw_heat_map_utility import draw_heat_map
from utils.draw_force_field_utility import draw_force_field

RAW_FRAME_DIMENSIONS = (320, 240)

def load_dataset(file_path: str):
    """Utility to load the force dataset .npz file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = np.load(file_path)
    # normal shape: (N, H, W, C), shear shape: (N, H, W, C), timestamps: (N,)
    return data['normal'], data['shear'], data['timestamps']


def plot_force_signals(file_path: str) -> None:
    """
    Utility 1: Plots the temporal evolution of the forces.
    Useful for studying the behavior of the sensor during contacts.
    """
    normal, shear, timestamps = load_dataset(file_path)
    
    # Calculate the relative time starting from 0 (in seconds)
    t = timestamps - timestamps[0]
    
    # Calculate the magnitude (norm) for each pixel along the last axis (the channels X, Y, Z)
    # If normal has a Z component, the norm coincides with the absolute value.
    normal_mag = np.linalg.norm(normal, axis=-1)
    shear_mag = np.linalg.norm(shear, axis=-1)
    
    # Calculate the mean force over the entire sensor surface for each frame
    mean_normal_over_time = np.mean(normal_mag, axis=(1, 2))
    mean_shear_over_time = np.mean(shear_mag, axis=(1, 2))
    
    # Calculate the maximum peak of the force for each frame (useful for detecting concentrated contacts)
    max_normal_over_time = np.max(normal_mag, axis=(1, 2))
    max_shear_over_time = np.max(shear_mag, axis=(1, 2))

    # Create the plots with an improved layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot Normal Force
    ax1.plot(t, mean_normal_over_time, label="Mean Surface", color="blue", linewidth=1.5)
    ax1.plot(t, max_normal_over_time, label="Maximum Peak", color="lightblue", alpha=0.7, linewidth=1.5)
    ax1.set_title("Temporal Evolution of Normal Force", fontsize=14)
    ax1.set_ylabel("Normal Magnitude [N]", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")
    
    # Plot Shear Force
    ax2.plot(t, mean_shear_over_time, label="Mean Surface", color="red", linewidth=1.5)
    ax2.plot(t, max_shear_over_time, label="Maximum Peak", color="salmon", alpha=0.7, linewidth=1.5)
    ax2.set_title("Temporal Evolution of Shear Force", fontsize=14)
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Shear Magnitude [N]", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()


def analyze_sensor_noise(file_path: str, raw_shape: tuple[int, int] = RAW_FRAME_DIMENSIONS) -> None:
    """
    Utility 2: Analyzes and visualizes the resting sensor offset and noise (standard deviation).
    """
    normal, shear, timestamps = load_dataset(file_path)
    
    print(f"\nAnalysis started on {len(timestamps)} frames (Duration: {timestamps[-1] - timestamps[0]:.2f} sec)")
    
    # =========================================================================
    # 1. OFFSET CALCULATION (Absolute Mean over time)
    # =========================================================================
    mean_normal = np.mean(normal, axis=0)
    mean_shear = np.mean(shear, axis=0)  # This keeps both the X and Y components
    
    mean_normal_mag = mean_normal[:, :, 0] if mean_normal.ndim == 3 else mean_normal
    mean_shear_mag = np.linalg.norm(mean_shear, axis=-1)
    
    # --- NORMAL OFFSET (Heatmap) ---
    normal_mean_bgr = draw_heat_map(mean_normal_mag, raw_shape=raw_shape, max_force=2.0)
    
    # --- SHEAR OFFSET (Vector Field / Arrows) ---
    # Using the custom utility by passing the vector means. 
    # force_bias=0.0 ensures we see even the smallest offset arrows.
    shear_mean_bgr = draw_force_field(mean_normal, mean_shear, raw_shape=raw_shape)
    
    # BGR -> RGB conversion for Matplotlib
    normal_mean_rgb = cv2.cvtColor(normal_mean_bgr, cv2.COLOR_BGR2RGB)
    shear_mean_rgb = cv2.cvtColor(shear_mean_bgr, cv2.COLOR_BGR2RGB)

    # =========================================================================
    # 2. NOISE CALCULATION (Standard Deviation over time)
    # =========================================================================
    normal_std_map = np.std(normal, axis=0)
    shear_std_map = np.std(shear, axis=0)
    
    normal_noise_magnitude = np.linalg.norm(normal_std_map, axis=-1) if normal_std_map.ndim == 3 else normal_std_map
    shear_noise_magnitude = np.linalg.norm(shear_std_map, axis=-1)
    
    cv2_target_size = (raw_shape[1], raw_shape[0]) 
    normal_noise_vis = cv2.resize(normal_noise_magnitude, cv2_target_size, interpolation=cv2.INTER_CUBIC)
    shear_noise_vis = cv2.resize(shear_noise_magnitude, cv2_target_size, interpolation=cv2.INTER_CUBIC)

    # =========================================================================
    # 3. GLOBAL METRICS
    # =========================================================================
    global_mean_normal_offset = np.mean(mean_normal_mag)
    global_mean_shear_offset = np.mean(mean_shear_mag)
    
    global_mean_normal_noise = np.mean(normal_noise_magnitude)
    global_mean_shear_noise = np.mean(shear_noise_magnitude)
    
    print("-" * 55)
    print("RESTING SENSOR METRICS")
    print(f"Normal Force -> OFFSET: {global_mean_normal_offset:.4f} N | NOISE: {global_mean_normal_noise:.4f} N")
    print(f"Shear Force  -> OFFSET: {global_mean_shear_offset:.4f} N | NOISE: {global_mean_shear_noise:.4f} N")
    print("-" * 55)
    
    # =========================================================================
    # 4. 2x2 GRAPHICAL VISUALIZATION
    # =========================================================================
    # Improved layout dimensions with explicit spacing
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={'wspace': 0.1, 'hspace': 0.25})
    
    # NORMAL - OFFSET
    axs[0, 0].imshow(normal_mean_rgb)
    axs[0, 0].set_title(f"NORMAL: Base Offset (Mean: {global_mean_normal_offset:.4f} N)", fontsize=12)
    axs[0, 0].axis("off")
    
    # NORMAL - NOISE (Grayscale mapping: white=0, dark gray=max)
    # Using 'gray_r' (reversed gray) colormap where 0 is white.
    im1 = axs[0, 1].imshow(normal_noise_vis, cmap="gray_r", vmin=0)
    axs[0, 1].set_title(f"NORMAL: Noise/Jitter (Mean: {global_mean_normal_noise:.4f} N)", fontsize=12)
    axs[0, 1].axis("off")
    cbar1 = fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label('Std Dev [N]')
    
    # SHEAR - OFFSET
    axs[1, 0].imshow(shear_mean_rgb)
    axs[1, 0].set_title(f"SHEAR: Offset Vectors (Mean Magnitude: {global_mean_shear_offset:.4f} N)", fontsize=12)
    axs[1, 0].axis("off")
    
    # SHEAR - NOISE (Grayscale mapping: white=0, dark gray=max)
    im2 = axs[1, 1].imshow(shear_noise_vis, cmap="gray_r", vmin=0)
    axs[1, 1].set_title(f"SHEAR: Noise/Jitter (Mean: {global_mean_shear_noise:.4f} N)", fontsize=12)
    axs[1, 1].axis("off")
    cbar2 = fig.colorbar(im2, ax=axs[1, 1], fraction=0.046, pad=0.04)
    cbar2.set_label('Std Dev [N]')
    
    plt.suptitle("Resting Sensor Analysis: Offset vs Noise", fontsize=18, fontweight="bold")
    
    # Prevent the main title from overlapping the subplots
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    # Insert your file name here!
    TEST_FILE = "../force_datasets/force_record_noise_study.npz"
    
    # plot_force_signals(TEST_FILE)
    analyze_sensor_noise(TEST_FILE)