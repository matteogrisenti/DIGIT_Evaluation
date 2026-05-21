import cv2
import numpy as np

def draw_force_field(normal_field, shear_field, raw_shape, force_bias=0.0, stride=14, arrow_scale=15.0):
    """
    Genera il campo vettoriale scientifico filtrando il bias di rumore statico.
    """
    if normal_field.ndim > 2:
        normal_field = np.squeeze(normal_field)
        
    h, w = normal_field.shape
    vis_image = np.full((h, w, 3), 255, dtype=np.uint8)
    
    dx_map = shear_field[:, :, 0]
    dy_map = shear_field[:, :, 1]
    magnitude_map = np.sqrt(dx_map**2 + dy_map**2)
    
    # Valore massimo grezzo registrato nel frame attuale
    mag_max_raw = magnitude_map.max()
    
    # Scala dinamica basata sul reale picco di forza registrato
    # Se non tocchi nulla, la scala si blocca su un valore di riferimento (es. 1.0) per evitare sfarfallii cromatici
    scale_max = max(mag_max_raw, 1.0)
    
    mag_norm = np.clip(magnitude_map / scale_max, 0, 1)
    color_palette = cv2.applyColorMap((mag_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    
    # La soglia di rumore è il nostro bias calibrato più un piccolo margine di tolleranza hardware
    noise_threshold = force_bias * 1.15
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            magnitude = magnitude_map[y, x]
            pt1 = (int(x), int(y))
            
            if magnitude < noise_threshold:
                # Sotto la soglia di calibrazione: solo puntini grigi a riposo
                cv2.circle(vis_image, pt1, 1, (180, 180, 180), -1, cv2.LINE_AA)
            else:
                # Sopra la soglia: disegna il vettore reale spinto dal dito
                dx = dx_map[y, x]
                dy = dy_map[y, x]
                
                pt2 = (int(x + dx * arrow_scale), int(y + dy * arrow_scale))
                arrow_color = color_palette[y, x].tolist()
                cv2.arrowedLine(vis_image, pt1, pt2, arrow_color, 1, cv2.LINE_AA, 0, 0.3)
                
    # Proporzionamento verticale
    target_h = raw_shape[0]
    target_w = raw_shape[1]
    vis_image_rect = cv2.resize(vis_image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Barra laterale
    bar_width = 75
    colorbar = np.full((target_h, bar_width, 3), 255, dtype=np.uint8)
    gradient = np.linspace(1, 0, target_h).reshape(-1, 1)
    gradient = np.repeat(gradient, 15, axis=1)
    gradient_colored = cv2.applyColorMap((gradient * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colorbar[:, 5:20] = gradient_colored
    
    # Se siamo sotto la soglia di rumore, forziamo il display a 0.00N
    display_force = mag_max_raw if mag_max_raw > noise_threshold else 0.00
    
    cv2.putText(colorbar, f"{display_force:.2f}N", (25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(colorbar, "0.00N", (25, target_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    return np.hstack((vis_image_rect, colorbar))