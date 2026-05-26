import cv2
import numpy as np

def draw_heat_map(
    normal_field, 
    raw_shape=None, 
    max_force=2.0, 
    colormap=cv2.COLORMAP_VIRIDIS, 
    blur_ksize=(21, 21), 
    bg_threshold=0.01
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
    normalized_threshold = bg_threshold / max_force  # Adjust background threshold to normalized scale
    
    # 5. INVERT THE SCALE: The reference uses Green for lowest forces and Blue for highest.
    # VIRIDIS puts 255 at Green and 0 at Blue, so we invert the normalized data.
    inverted_normalized = 1.0 - normal_normalized
    normal_uint8 = (inverted_normalized * 255).astype(np.uint8)
    
    # 6. Apply the OpenCV color map
    heatmap_view = cv2.applyColorMap(normal_uint8, colormap)
    
    # 7. WHITE BACKGROUND MASK: Find areas with near-zero force and paint them white
    bg_mask = normal_normalized < normalized_threshold
    heatmap_view[bg_mask] = [255, 255, 255]
    
    # 8. Resize to match the target camera shape
    if raw_shape is not None:
        target_size = (raw_shape[1], raw_shape[0]) 
        if (heatmap_view.shape[1], heatmap_view.shape[0]) != target_size:
            heatmap_view = cv2.resize(heatmap_view, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 9. Add a vertical colorbar on the right side for reference
    target_h = heatmap_view.shape[0]
    bar_width = 75
    colorbar = np.full((target_h, bar_width, 3), 255, dtype=np.uint8)
            
    # Create a vertical gradient for the colorbar
    gradient = np.linspace(0, 1, target_h).reshape(-1, 1)
    gradient = np.repeat(gradient, 15, axis=1)
    gradient_colored = cv2.applyColorMap((gradient * 255).astype(np.uint8), colormap)
    colorbar[:, 5:20] = gradient_colored
    
    # Annotate the colorbar with force values
    cv2.putText(colorbar, f"{max_force:.2f}N", (25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(colorbar, "0.00N", (25, target_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    return np.hstack((heatmap_view, colorbar))