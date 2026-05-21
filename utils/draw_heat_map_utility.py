import cv2
import numpy as np

def draw_heat_map(normal_field, raw_shape=None, max_force=2.0, colormap=cv2.COLORMAP_TURBO):
    """
    Generates a colored heatmap from the normal force field predictions.
    
    Args:
        normal_field (np.ndarray): The predicted normal forces (H, W, 1) or (H, W).
        raw_shape (tuple): The target spatial resolution (Height, Width, Channels) to match the camera.
        max_force (float): The maximum force value (in Newtons) to saturate the heatmap colors (Red).
        colormap (int): OpenCV colormap flag. Default is TURBO (smooth depth gradient).
        
    Returns:
        np.ndarray: The colored heatmap image.
    """
    # 1. Extract the 2D array if it has an extra channel dimension
    if normal_field.ndim == 3:
        normal_mag = normal_field[:, :, 0]
    else:
        normal_mag = normal_field
        
    # 2. Clean the data: clip negative noise (we only care about inward pressure)
    normal_mag = np.clip(normal_mag, 0, None)
    
    # 3. Normalize the forces to a 0.0 - 1.0 scale
    normal_normalized = np.clip(normal_mag / max_force, 0.0, 1.0)
    
    # 4. Convert to an 8-bit image format (0-255) for OpenCV
    normal_uint8 = (normal_normalized * 255).astype(np.uint8)
    
    # 5. Apply the OpenCV color map
    heatmap_view = cv2.applyColorMap(normal_uint8, colormap)
    
    # 6. Resize to match the target shape if provided
    if raw_shape is not None:
        # cv2 shape is (Height, Width, Channels), but cv2.resize expects (Width, Height)
        target_size = (raw_shape[1], raw_shape[0]) 
        if (heatmap_view.shape[1], heatmap_view.shape[0]) != target_size:
            heatmap_view = cv2.resize(heatmap_view, target_size, interpolation=cv2.INTER_LINEAR)
            
    return heatmap_view