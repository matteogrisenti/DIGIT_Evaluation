import cv2
import numpy as np

def draw_heat_map(
    normal_field, 
    raw_shape=None, 
    max_force=2.0, 
    colormap=cv2.COLORMAP_INFERNO, 
    blur_ksize=(21, 21), 
    bg_threshold=0.05
):
    """
    Generates a smoothed, professional heatmap resembling the original Sparsh reference.
    
    Args:
        normal_field (np.ndarray): The predicted normal forces (H, W, 1) or (H, W).
        raw_shape (tuple): Target spatial resolution (Height, Width, Channels).
        max_force (float): Max force in Newtons. 
        colormap (int): OpenCV colormap. INFERNO inverted gives the Yellow -> Black scale.
        blur_ksize (tuple): Gaussian Blur kernel size to remove ViT patch grid artifacts.
        bg_threshold (float): Force threshold below which the background is painted pure white.
    """
    # 1. Extract the 2D array if it has an extra channel dimension
    if normal_field.ndim == 3:
        normal_mag = normal_field[:, :, 0]
    else:
        normal_mag = normal_field
        
    # 2. SMOOTHING: Apply Gaussian Blur to melt the ViT patch grid into a smooth field
    normal_mag = cv2.GaussianBlur(normal_mag, blur_ksize, 0)
        
    # 3. Clean the data: clip negative noise
    normal_mag = np.clip(normal_mag, 0, None)
    
    # 4. Normalize the forces to a 0.0 - 1.0 scale based on the max_force
    normal_normalized = np.clip(normal_mag / max_force, 0.0, 1.0)
    
    # 5. INVERT THE SCALE: The reference uses Yellow for lowest forces and Black for highest.
    # INFERNO puts 255 at Yellow and 0 at Black, so we invert the normalized data.
    inverted_normalized = 1.0 - normal_normalized
    normal_uint8 = (inverted_normalized * 255).astype(np.uint8)
    
    # 6. Apply the OpenCV color map
    heatmap_view = cv2.applyColorMap(normal_uint8, colormap)
    
    # 7. WHITE BACKGROUND MASK: Find areas with near-zero force and paint them white
    bg_mask = normal_normalized < bg_threshold
    heatmap_view[bg_mask] = [255, 255, 255]
    
    # 8. Resize to match the target camera shape
    if raw_shape is not None:
        target_size = (raw_shape[1], raw_shape[0]) 
        if (heatmap_view.shape[1], heatmap_view.shape[0]) != target_size:
            heatmap_view = cv2.resize(heatmap_view, target_size, interpolation=cv2.INTER_LINEAR)
            
    return heatmap_view