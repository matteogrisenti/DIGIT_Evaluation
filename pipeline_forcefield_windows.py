import os
import sys
import collections
import cv2
import numpy as np
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from hydra.utils import instantiate

import unittest.mock as mock
if 'xformers' not in sys.modules:
    sys.modules['xformers'] = mock.MagicMock()
    sys.modules['xformers.ops'] = mock.MagicMock()
    # Aggiungiamo anche fmha per sicurezza, visto l'errore precedente
    sys.modules['xformers.ops.fmha'] = mock.MagicMock()

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
    
    # 1. Carica i modelli PyTorch
    encoder, decoder = load_sparsh_models(CHECKPOINTS_DIR, DEVICE)
    
    # 2. Connessione al sensore DIGIT usando il tuo backend Windows (indice camera 0, 1, ecc.)
    # print("Inizializzazione del sensore DIGIT su Windows...")
    # digit_sensor = Digit(device_index=0, name="Windows_DIGIT")
    # digit_sensor.connect()
    
    # # 3. Acquisizione del Frame di Background (Il gel deve essere a riposo, senza contatto!)
    # print("Acquisizione frame di background in corso... Non toccare il sensore.")
    # # Attendiamo qualche frame per stabilizzare l'esposizione automatica della camera
    # for _ in range(10):
    #     _ = digit_sensor.get_frame()
    # bg_frame = digit_sensor.get_frame()
    # print("Background acquisito con successo!")
    
    # # 4. Buffer Temporale: Sparsh richiede la storia recente per stimare lo slittamento e le forze di taglio.
    # # Usiamo una coda a dimensione fissa per memorizzare gli ultimi 6 frame pre-processati.
    # # Questo ci permette di recuperare istantaneamente I_t e I_{t-5}
    # frame_history = collections.deque(maxlen=6)
    
    # print("\nPipeline avviata! Premi 'ESC' sulla finestra video per uscire.")
    
    # try:
    #     while True:
    #         # Cattura il frame corrente dal sensore Windows
    #         raw_frame = digit_sensor.get_frame()
            
    #         # Applica pre-processing (Sottrazione BG + Resize 224x224)
    #         processed_frame = pre_process_frame(raw_frame, bg_frame)
    #         frame_history.append(processed_frame)
            
    #         # Se il buffer non è ancora pieno (servono almeno 6 frame), saltiamo l'inferenza
    #         if len(frame_history) < 6:
    #             continue
                
    #         # Recuperiamo I_t e I_{t-5} per creare la finestra di inferenza di ~80ms
    #         I_t = frame_history[-1]      # Ultimo inserito
    #         I_t_minus_5 = frame_history[0] # Primo inserito (5 passi indietro)
            
    #         # Concatena i due frame lungo la dimensione dei canali (H, W, 3+3 = 6)
    #         input_channels = np.concatenate([I_t, I_t_minus_5], axis=-1) 
            
    #         # Adatta la forma per PyTorch da [H, W, C] a [B, C, H, W] -> [1, 6, 224, 224]
    #         input_tensor = torch.from_numpy(input_channels).permethod((2, 0, 1)).unsqueeze(0).to(DEVICE)
            
    #         # --- INFERENZA PYTORCH (Frozen Encoder) ---
    #         with torch.no_grad():
    #             # Passaggio nell'encoder DINO (estrazione feature layer intermedi 2, 5, 8, 11)
    #             # intermediate_features = encoder.get_intermediate_layers(input_tensor, n=[2, 5, 8, 11])
                
    #             # Passaggio nel decoder DPT per la stima densa delle forze
    #             # normal_field, shear_field = decoder(intermediate_features)
                
    #             # Mock output per visualizzazione temporanea
    #             normal_field_mock = np.zeros((224, 224, 1))
    #             shear_field_mock = np.zeros((224, 224, 2))
            
    #         # 5. Rendering del Campo di Forze
    #         output_view = draw_force_field(normal_field_mock, shear_field_mock)
            
    #         # Mostra la vista live ed il feed della camera originale
    #         cv2.imshow("DIGIT Raw Frame (Windows)", raw_frame)
    #         cv2.imshow("Sparsh Force Field Visualization", output_view)
            
    #         # Interrompi se l'utente preme ESC
    #         if cv2.waitKey(1) == 27:
    #             break
                
    # except Exception as e:
    #     print(f"Errore durante l'esecuzione della pipeline: {e}")
        
    # finally:
    #     # Chiusura pulita del sensore
    #     print("Disconnessione dal sensore DIGIT...")
    #     digit_sensor.disconnect()
    #     cv2.destroyAllWindows()

if __name__ == "__main__":
    main()