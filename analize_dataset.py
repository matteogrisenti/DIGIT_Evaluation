import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

RAW_FRAME_DIMENSIONS = (320, 240)

def load_dataset(file_path: str):
    """Utility to load the force dataset file .npz"""
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

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Normal Force
    ax1.plot(t, mean_normal_over_time, label="Mean Surface", color="blue")
    ax1.plot(t, max_normal_over_time, label="Maximum Peak", color="lightblue", alpha=0.7)
    ax1.set_title("Temporal Evolution of Normal Force")
    ax1.set_ylabel("Normal Magnitude")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    # Plot Shear Force
    ax2.plot(t, mean_shear_over_time, label="Mean Surface", color="red")
    ax2.plot(t, max_shear_over_time, label="Maximum Peak", color="salmon", alpha=0.7)
    ax2.set_title("Temporal Evolution of Shear Force")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Shear Magnitude")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def analyze_sensor_noise(file_path: str, raw_shape: tuple[int, int] = RAW_FRAME_DIMENSIONS ) -> None:
    """
    Utility 2: Studies the noise characteristics of the sensor by analyzing the variance of the force measurements over time.
    This function should be used on a dataset recorded WITHOUT touching the gel.
    """
    normal, shear, timestamps = load_dataset(file_path)
    
    print(f"Noise analysis started on {len(timestamps)} frames ({timestamps[-1] - timestamps[0]:.2f} sec)")
    
    # 1. Standard Deviation Over Time: 
    # For each pixel, calculate the std of the force measurements across all frames.
    normal_std_map = np.std(normal, axis=0)
    shear_std_map = np.std(shear, axis=0)
    
    # Normalize the noise maps to get a single "noise magnitude" value for each pixel 
    # (useful for visualization)
    normal_noise_magnitude = np.linalg.norm(normal_std_map, axis=-1)
    shear_noise_magnitude = np.linalg.norm(shear_std_map, axis=-1)
    
    # 2. Global Noise Metrics: 
    # Calculate the mean and max noise magnitude across the entire sensor to get a sense of overall noise levels.
    global_mean_normal_noise = np.mean(normal_noise_magnitude)
    global_max_normal_noise = np.max(normal_noise_magnitude)
    
    global_mean_shear_noise = np.mean(shear_noise_magnitude)
    global_max_shear_noise = np.max(shear_noise_magnitude)
    
    print("-" * 40)
    print("METRICHE DI RUMORE GLOBALI (Deviazione Std media nel tempo)")
    print(f"Forza Normale -> Rumore Medio: {global_mean_normal_noise:.5f} | Picco: {global_max_normal_noise:.5f}")
    print(f"Forza Taglio  -> Rumore Medio: {global_mean_shear_noise:.5f}  | Picco: {global_max_shear_noise:.5f}")
    print("-" * 40)
    
    # --- RESIZE PER LA VISUALIZZAZIONE RETTANGOLARE ---
    # Invertiamo (Altezza, Larghezza) di NumPy in (Larghezza, Altezza) per OpenCV
    cv2_target_size = (raw_shape[1], raw_shape[0]) 
    
    # Applichiamo un'interpolazione cubica per avere una mappa sfumata e pulita
    normal_noise_vis = cv2.resize(normal_noise_magnitude, cv2_target_size, interpolation=cv2.INTER_CUBIC)
    shear_noise_vis = cv2.resize(shear_noise_magnitude, cv2_target_size, interpolation=cv2.INTER_CUBIC)
    
    # 3. Visualizzazione spaziale del rumore (Heatmaps)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(normal_noise_vis, cmap="viridis")
    ax1.set_title(f"Mappa Rumore Normale (Media: {global_mean_normal_noise:.4f})")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    im2 = ax2.imshow(shear_noise_vis, cmap="plasma")
    ax2.set_title(f"Mappa Rumore Shear (Media: {global_mean_shear_noise:.4f})")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.suptitle("Analisi Spaziale del Rumore del Sensore", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Sostituisci con il nome del tuo file appena generato!
    TEST_FILE = "force_datasets/force_record_20260526_125851.npz"
    
    # De-commenta una delle due funzioni in base a ciò che vuoi studiare:
    
    # 1. Per studiare i contatti nel tempo:
    plot_force_signals(TEST_FILE)
    
    # 2. Per valutare il rumore (assicurati che il dataset sia a vuoto/riposo):
    analyze_sensor_noise(TEST_FILE)
    pass