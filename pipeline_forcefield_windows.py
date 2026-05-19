import os
import sys
import collections
import cv2
import numpy as np
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from hydra.utils import instantiate


# 1. Configurazione dei Path: Aggiungiamo il sottomodulo sparsh al path di Python
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SPARSH_SUBMODULE_PATH = os.path.join(CURRENT_DIR, "sparsh")
if SPARSH_SUBMODULE_PATH not in sys.path:
    sys.path.append(SPARSH_SUBMODULE_PATH)

# Importiamo la TUA classe Digit per Windows
from digit_interface.digit_windows import Digit

# Importa i modelli Sparsh
from sparsh.tactile_ssl.model.vision_transformer import vit_base


def load_sparsh_models(checkpoints_dir, device):
    """
    Upload of the DINO Encoder and DPT Decoder models to the specified device (GPU/CPU).
    """
    
    print(f"Uploading models to device: {device}")
    
    # --- 1. Initialization of Encoder DINO (ViT-Base) ---
    # Sparsh DINO Base use a ViT-Base with patch of 14 and 6 channel in input
    encoder = vit_base(
        img_size=224,
        in_chans=6,            # 6 channels: 3 for I_t and 3 for I_{t-5} 
        patch_size=14,         # Sparsh DINO Base uses patch size of 14
        num_register_tokens=1  
    )
    
    # Upload encoder weights
    encoder_path = os.path.join(checkpoints_dir, "encoder_dino", "dino_vitbase.ckpt")
    print("Uploading encoder weights...")

    # Load the .ckpt file
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    
    # Extract the state_dict from the checkpoint. Depending on how it was saved.
    if "state_dict" in checkpoint:
        encoder_state_dict = checkpoint["state_dict"]
    else:
        encoder_state_dict = checkpoint 
    
    # Clean the state_dict keys if they have unwanted prefixes (e.g., 'backbone.' or 'model_task.')
    if "backbone" in list(encoder_state_dict.keys())[0]:
        encoder_state_dict = {k.replace("backbone.", ""): v for k, v in encoder_state_dict.items()}
        
    encoder.load_state_dict(encoder_state_dict, strict=False)
    encoder.to(device)
    encoder.eval()

    import unittest.mock as mock
    if 'xformers' not in sys.modules:
        sys.modules['xformers'] = mock.MagicMock()
        sys.modules['xformers.ops'] = mock.MagicMock()
        sys.modules['xformers.ops.fmha'] = mock.MagicMock()
    
    # --- 2. Initialization of Decoder DPT (Force Field) ---
    print("Instantiating decoder model...")
    
    config_path = os.path.join(
        SPARSH_SUBMODULE_PATH, 
        "config", "experiment", "downstream_task", "forcefield", "digit_dino.yaml"
    )
    
    # Load the decoder configuration using OmegaConf
    cfg = OmegaConf.load(config_path)
    
    # Extract the decoder configuration from the loaded config
    decoder_cfg = cfg.task.model_task
    
    # Using Hydra's instantiate to create the decoder model based on the configuration
    decoder = instantiate(decoder_cfg)
    
    # Upload decoder weights
    decoder_path = os.path.join(checkpoints_dir, "forcefield_decoder", "digit_t1_forcefield_dino_vitbase_bg", "checkpoints", "last.ckpt")
    print("Uploading decoder weights...")
    
    # Load the .ckpt file
    decoder_ckpt = torch.load(decoder_path, map_location=device, weights_only=False)
    
    # Extract the state_dict from the checkpoint. Depending on how it was saved.
    if "state_dict" in decoder_ckpt:
        decoder_state_dict = decoder_ckpt["state_dict"]
    else:
        decoder_state_dict = decoder_ckpt
    
    # Clean the state_dict keys if they have unwanted prefixes (e.g., 'backbone.' or 'model_task.')
    clean_decoder_dict = {}
    for k, v in decoder_state_dict.items():
        new_k = k.replace("model_task.", "").replace("module.", "")
        clean_decoder_dict[new_k] = v
        
    # Upload the cleaned state_dict into the decoder model
    decoder.load_state_dict(clean_decoder_dict, strict=False)
    
    decoder.to(device)
    decoder.eval()
    
    print(f"Encoder e Decoder uploaded successfully on {device}")
    return encoder, decoder


def pre_process_frame(frame, bg_frame):
    """
    Esegue il pre-processing dell'immagine tattile secondo le specifiche di Sparsh.
    """
    # 1. Sottrazione del background per garantire la generalizzazione tra sensori
    frame_diff = cv2.subtract(frame, bg_frame)
    
    # 2. Conversione in float32 e normalizzazione [0, 1]
    frame_float = frame_diff.astype(np.float32) / 255.0
    
    # 3. Ridimensionamento alla risoluzione nativa richiesta dal ViT (224x224)
    frame_resized = cv2.resize(frame_float, (224, 224))
    
    return frame_resized


def draw_force_field(normal_field, shear_field):
    """
    Prende l'output denso del decoder DPT e genera l'immagine del campo di forze.
    Le frecce rappresentano la forza di taglio, il colore lo schiacciamento normale.
    """
    # normal_field: mappa di profondità densa (es. 224x224x1) [cite: 168]
    # shear_field: mappa del flusso ottico densa (es. 224x224x2) [cite: 168]
    
    # Creiamo un canvas per la visualizzazione
    vis_image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # TODO: Logica di disegno vettoriale OpenCV (cv2.arrowedLine) 
    # Mappa le forze normali sui canali colore e disegna le frecce di taglio ad intervalli regolari (stride es. 10px) [cite: 763]
    
    return vis_image

def main():
    # Definiamo i percorsi dei checkpoint
    CHECKPOINTS_DIR = os.path.join(CURRENT_DIR, "outputs_sparsh", "checkpoints")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device detected and used: {DEVICE}")
    
    # 1. Upload DINO Encoder e DPT Decoder on device (GPU/CPU)
    encoder, decoder = load_sparsh_models(CHECKPOINTS_DIR, DEVICE)
    
    # 2. Connection to DIGIT Sensor using custom Windows driver
    print("Initializing connection to DIGIT sensor on Windows...")
    digit_sensor = Digit(device_index=0, name="Windows_DIGIT")
    digit_sensor.connect()
    
    # 3. Background Acquisition: Sparsh relies on background subtraction for robust tactile perception. 
    # We capture a stable background frame at the start of the pipeline to use for pre-processing all subsequent frames.
    print("Acquiring background frame for subtraction... Please ensure the sensor is not in contact with any object.")
    # wait a few frames to stabilize the sensor and then capture the background
    for _ in range(10):
        _ = digit_sensor.get_frame()
    bg_frame = digit_sensor.get_frame()
    print("Background frame acquired successfully.")
    # # DEBUG: saving the background frame for inspection
    # cv2.imwrite("bg_frame_windows.png", bg_frame)
    
    # 4. Temporal Buffer: Sparsh encoder require two frame in input: 
    #      - the one at time t 
    #      - the one at time t - 5.
    # We use a deque with maxlen=6 to keep the last 6 frames.
    # This allows us to collect the I_t (last) e I_{t-5} (first) frame
    frame_history = collections.deque(maxlen=6)
    
    print("\nPipeloine started! Press 'ESC' on the video window to exit.")
    
    try:
        while True:
            # capture a new frame from the DIGIT sensor
            raw_frame = digit_sensor.get_frame()
            
            # Applay Preprocessing (background subtraction, normalization, resizing) 
            processed_frame = pre_process_frame(raw_frame, bg_frame)
            frame_history.append(processed_frame)
            
            # If the history buffer is not full yet, we cannot perform inference 
            # (we need at least 6 frames to have I_t and I_{t-5})
            if len(frame_history) < 6:
                continue
                
            # Take the last frame (I_t) and the frame from 5 steps back (I_{t-5}) 
            I_t = frame_history[-1]         # Last inserted (current frame)
            I_t_minus_5 = frame_history[0]  # First inserted (5 steps back)
            
            # Concatena i due frame lungo la dimensione dei canali (H, W, 3+3 = 6)
            input_channels = np.concatenate([I_t, I_t_minus_5], axis=-1) 
            
            # Abdate the shape for PyTorch from [H, W, C] to [B, C, H, W] -> [1, 6, 224, 224]
            input_tensor = torch.from_numpy(input_channels).permute((2, 0, 1)).unsqueeze(0).to(DEVICE)
            
            # --- INFERENZA PYTORCH (Frozen Encoder) ---
            with torch.no_grad():
                # 1. Estraiamo la tupla di tensori dall'encoder
                intermediate_features = encoder.get_intermediate_layers(
                    input_tensor, 
                    n=[2, 5, 8, 11], 
                )
                
                # 2. TRUCCO: Trasformiamo la tupla nel Dizionario che il Decoder si aspetta!
                # Diamo delle chiavi fittizie (i nomi dei layer) in ordine.
                features_dict = {
                    "t2": intermediate_features[0],
                    "t5": intermediate_features[1],
                    "t8": intermediate_features[2],
                    "t11": intermediate_features[3]
                }
                
                # 3. Passiamo il dizionario al decoder
                force_field_pred = decoder(features_dict)
                
                # 4. Estrazione dei canali di output e conversione per OpenCV
                if isinstance(force_field_pred, dict):
                    normal_tensor = force_field_pred.get("normal", force_field_pred.get("depth"))
                    shear_tensor = force_field_pred.get("shear", force_field_pred.get("flow"))
                elif isinstance(force_field_pred, (tuple, list)):
                    normal_tensor = force_field_pred[0]
                    shear_tensor = force_field_pred[1]
                else:
                    normal_tensor = force_field_pred[:, 0:1, :, :]
                    shear_tensor = force_field_pred[:, 1:3, :, :]

                # Trasformazione per OpenCV
                normal_field = normal_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                shear_field = shear_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # 5. Rendering del Campo di Forze
            output_view = draw_force_field(normal_field, shear_field)
            
            # Mostra la vista live ed il feed della camera originale
            raw_frame_pulito = np.ascontiguousarray(raw_frame)
            cv2.imshow("DIGIT Raw Frame (Windows)", raw_frame_pulito)
            cv2.imshow("Sparsh Force Field Visualization", output_view)
            
            # Interrompi se l'utente preme ESC
            if cv2.waitKey(1) == 27:
                break
                
    except Exception as e:
        import traceback
        print("\n" + "!"*60)
        print("ERRORE CRITICO NELLA PIPELINE - ECCO I DETTAGLI:")
        traceback.print_exc()
        print("!"*60 + "\n")
        
    finally:
        print("Disconnessione dal sensore DIGIT...")
        digit_sensor.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()