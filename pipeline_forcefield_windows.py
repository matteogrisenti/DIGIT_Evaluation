import os
import sys
import collections
import cv2
import numpy as np
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from hydra.utils import instantiate

from utils.draw_force_field_utility import draw_force_field


# ==============================================================================
# 1. TRUCCO DI BYPASS HARDWARE (MOCK XFORMERS PER WINDOWS)
# ==============================================================================
import unittest.mock as mock

# Creiamo una struttura a cipolla finta per simulare xformers prima degli import di Sparsh
mock_xformers = mock.MagicMock()
mock_ops = mock.MagicMock()
mock_fmha = mock.MagicMock()

mock_ops.fmha = mock_fmha
mock_xformers.ops = mock_ops

sys.modules['xformers'] = mock_xformers
sys.modules['xformers.ops'] = mock_ops
sys.modules['xformers.ops.fmha'] = mock_fmha


# 1. Configurazione dei Path: Aggiungiamo il sottomodulo sparsh al path di Python
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SPARSH_SUBMODULE_PATH = os.path.join(CURRENT_DIR, "sparsh")
if SPARSH_SUBMODULE_PATH not in sys.path:
    sys.path.append(SPARSH_SUBMODULE_PATH)

# Importiamo la TUA classe Digit per Windows
from digit_interface.digit_windows import Digit

# Importa i modelli Sparsh
from sparsh.tactile_ssl.model.vision_transformer import vit_base
from sparsh.tactile_ssl.downstream_task.forcefield_sl import ForceFieldDecoder


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
    
    # --- 2. Inizializzazione Decoder DPT ---
    decoder = ForceFieldDecoder()
    
    # Carichiamo il file del checkpoint combinato
    decoder_path = os.path.join(checkpoints_dir, "forcefield_decoder", "digit_t1_forcefield_dino_vitbase_bg", "checkpoints","last.ckpt")
    checkpoint = torch.load(decoder_path, map_location=device, weights_only=False)
    
    # Estraiamo il dizionario globale "model" scoperto dal test
    if "model" in checkpoint:
        global_weights = checkpoint["model"]
    else:
        raise Exception("Errore critico: Impossibile trovare la macro-chiave 'model' nel checkpoint!")
        
    # SEPARAZIONE CHIRURGICA DEI PESI
    clean_decoder_dict = {}
    for k, v in global_weights.items():
        # Se la chiave appartiene al decoder (inizia con model_task.)
        if k.startswith("model_task."):
            # Rimuoviamo il prefisso 'model_task.' per farlo combaciare col nostro modello locale
            # Es: 'model_task.reassembles.0...' diventa 'reassembles.0...'
            new_k = k.replace("model_task.", "")
            clean_decoder_dict[new_k] = v

    # CARICAMENTO PARAMETRI NEL DECODER
    # Usiamo strict=True per essere sicuri che ogni singolo peso combaci al 100%
    try:
        decoder.load_state_dict(clean_decoder_dict, strict=True)
        print("--> [SUCCESS] Decoder caricato e sincronizzato al 100% con strict=True!")
    except RuntimeError as e:
        print("\n" + "!"*60)
        print("Mismatch parziale nelle chiavi della testa predittiva.")
        print("Caricamento forzato in modalità adattiva (strict=False).")
        print("!"*60)
        decoder.load_state_dict(clean_decoder_dict, strict=False)

    decoder.to(device)
    decoder.eval()
    
    print("Tutti i modelli sono stati caricati e sono pronti per l'inferenza.")
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

    # --- NUOVO: CALIBRAZIONE DELLO ZERO DELLA RETE ---
    print("Calibrazione dello zero del campo di forze... Attendi...")
    zero_magnitudes = []

    # --- CALIBRAZIONE DINAMICA DELLO ZERO ---
    print("Calibrazione dello zero del campo di forze (Movimento a vuoto)...")
    zero_magnitudes = []
    calib_history = collections.deque(maxlen=6)
    
    with torch.no_grad():
        for _ in range(25): # Facciamo girare il buffer per stabilizzarlo
            raw_f = digit_sensor.get_frame()
            proc_f = pre_process_frame(raw_f, bg_frame)
            calib_history.append(proc_f)
            
            if len(calib_history) == 6:
                I_t = calib_history[-1]
                I_t_minus_5 = calib_history[0]
                input_chan = np.concatenate([I_t, I_t_minus_5], axis=-1)
                in_tensor = torch.from_numpy(input_chan).permute((2, 0, 1)).unsqueeze(0).to(DEVICE)
                
                inter_feat = encoder.get_intermediate_layers(in_tensor, n=[2, 5, 8, 11])
                f_dict = {"t2": inter_feat[0], "t5": inter_feat[1], "t8": inter_feat[2], "t11": inter_feat[3]}
                pred = decoder(f_dict)
                
                if isinstance(pred, dict):
                    sh_tensor = pred.get("shear", pred.get("flow"))
                else:
                    sh_tensor = pred[:, 1:3, :, :]
                    
                sh_field = sh_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                mag_map = np.sqrt(sh_field[:, :, 0]**2 + sh_field[:, :, 1]**2)
                zero_magnitudes.append(mag_map.max())
                
    FORCE_BIAS = np.mean(zero_magnitudes) if len(zero_magnitudes) > 0 else 0.0
    print(f"Calibrazione completata. Bias reale rilevato: {FORCE_BIAS:.4f}")
    
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
            output_view = draw_force_field(
                normal_field, 
                shear_field, 
                raw_shape=raw_frame.shape,
                force_bias=0.0,  # Applichiamo il bias di forza calibrato per ottenere un campo più realistico
                stride=20,        # Cambia questo valore per avere una griglia più densa (es. 10) o più rada (es. 20)
                arrow_scale=10.0  # Cambia questo per allungare/accorciare le frecce
            )
            
            # Mostra la vista live ed il feed della camera originale
            cv2.imshow("DIGIT Raw Frame (Windows)", raw_frame)
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