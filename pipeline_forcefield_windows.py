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
from utils.draw_heat_map_utility import draw_heat_map


from utils.xformers_mock import patch_xformers
patch_xformers()


# 1. Configurazione dei Path: Aggiungiamo il sottomodulo sparsh al path di Python
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SPARSH_SUBMODULE_PATH = os.path.join(CURRENT_DIR, "sparsh")
if SPARSH_SUBMODULE_PATH not in sys.path:
    sys.path.append(SPARSH_SUBMODULE_PATH)

# Importiamo la TUA classe Digit per Windows
from digit_interface.digit_windows import Digit

# Importa i modelli Sparsh
from sparsh.tactile_ssl.model.vision_transformer import vit_base
from sparsh.tactile_ssl.downstream_task.forcefield_sl import ForceFieldModule


def load_sparsh_models(checkpoints_dir, device):
    """
    Carica l'Encoder e il Decoder usando le configurazioni originali (Hydra),
    li unisce in ForceFieldModule e carica i pesi unificati.
    """
    print(f"Uploading models to device: {device}")
    
    # --- 1. Initialization of Encoder DINO (ViT-Base) come da tuo vecchio file ---
    print("Instantiating encoder model...")
    encoder = vit_base(
        img_size=224,
        in_chans=6,            # 6 channels: 3 for I_t and 3 for I_{t-5} 
        patch_size=16,         
        num_register_tokens=1  # Recuperato dal tuo codice originale!
    )
    
    # --- 2. Initialization of Decoder DPT (Force Field) via Hydra ---
    print("Instantiating decoder model via Hydra config...")
    config_path = os.path.join(
        SPARSH_SUBMODULE_PATH, 
        "config", "experiment", "downstream_task", "forcefield", "digit_dino.yaml"
    )
    cfg = OmegaConf.load(config_path)
    decoder_cfg = cfg.task.model_task
    decoder = instantiate(decoder_cfg)
    
    # --- 3. Creazione del Modulo Contenitore (ForceFieldModule) ---
    print("Wrapping Encoder and Decoder in ForceFieldModule...")
    # Creiamo un finto ssl_config per soddisfare il costruttore della classe senza crash
    dummy_ssl_config = {
        "img_sz": (224, 224),
        "loss": {
            "with_sl_supervision": False,
            "with_mask_supervision": False,
            "with_ssim": False,
            "min_depth": 0.1,
            "max_depth": 10.0
        },
        "pose_estimator": {"num_encoder_layers": 18}
    }
    
    model_module = ForceFieldModule(
        model_encoder=encoder,
        model_task=decoder,
        optim_cfg=None,
        scheduler_cfg=None,
        ssl_config=dummy_ssl_config
    )
    
    # --- 4. Caricamento dei Pesi Unificati (Il file .ckpt combinato) ---
    print("Uploading unified weights...")
    # Sostituisci questo percorso se il nome del file che abbiamo ispezionato è in una cartella diversa
    checkpoint_path = os.path.join(checkpoints_dir, "forcefield_decoder", "digit_t1_forcefield_dino_vitbase_bg", 
                                   "checkpoints", "last.ckpt")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model" in checkpoint:
        global_weights = checkpoint["model"]
    elif "state_dict" in checkpoint:
        global_weights = checkpoint["state_dict"]
    else:
        global_weights = checkpoint
        
    # Carichiamo i pesi (strict=False perché ignoriamo la testa di PoseEstimator non usata)
    model_module.load_state_dict(global_weights, strict=False)
    
    model_module.to(device)
    model_module.eval()
    
    print(f"ForceFieldModule uploaded successfully on {device}")
    return model_module


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
    CHECKPOINTS_DIR = os.path.join(CURRENT_DIR, "outputs_sparsh", "checkpoints")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device rilevato: {DEVICE}")
    
    # 1. Caricamento del modulo contenitore completo
    model_module = load_sparsh_models(CHECKPOINTS_DIR, DEVICE)
    
    # 2. Connessione hardware al sensore DIGIT
    print("Inizializzazione connessione DIGIT su Windows...")
    digit_sensor = Digit(device_index=0, name="Windows_DIGIT")
    digit_sensor.connect()
    
    print("Acquisisco il background... Non toccare il gel...")
    for _ in range(10):
        _ = digit_sensor.get_frame()
    bg_frame = digit_sensor.get_frame()
    print("Background acquisito con successo.")

    # 3. Buffer temporale per gestire lo Stride di 5 frame
    frame_history = collections.deque(maxlen=6)
    last_inserted_frame = None
    
    print("\nPipeline avviata! Premi 'ESC' per uscire.")
    
    try:
        while True:
            raw_frame = digit_sensor.get_frame()
            processed_frame = pre_process_frame(raw_frame, bg_frame)
            
            # Filtro anti-clonazione del frame
            if last_inserted_frame is not None and np.array_equal(processed_frame, last_inserted_frame):
                cv2.waitKey(1)
                continue
                
            frame_history.append(processed_frame)
            last_inserted_frame = processed_frame
            
            if len(frame_history) < 6:
                continue
                
            # Estrazione della coppia temporale (I_t e I_{t-5})
            I_t = frame_history[-1]         
            I_t_minus_5 = frame_history[0]  
            
            input_channels = np.concatenate([I_t, I_t_minus_5], axis=-1) 
            input_tensor = torch.from_numpy(input_channels).permute((2, 0, 1)).unsqueeze(0).to(DEVICE)
            
            # --- INFERENZA DIRETTA CON FORCEFIELDMODULE ---
            with torch.no_grad():
                # Chiamiamo direttamente il forward del modulo completo!
                # Restituisce un dizionario contenente le stime dense "normal" e "shear"
                force_field_pred = model_module(input_tensor)
                
                # Estrazione dei tensori predetti
                normal_tensor = force_field_pred["normal"]
                shear_tensor = force_field_pred["shear"]

                # Trasformazione finale per OpenCV
                normal_field = normal_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                shear_field = shear_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # 4. Rendering grafico del campo di forze
            output_view = draw_force_field(
                normal_field, 
                shear_field, 
                raw_shape=raw_frame.shape,
                force_bias=0.0,    # Lasciamo a 0.0 per analizzare la dinamica pura dei pesi reali
                stride=14,         # Dimensione ottimale della griglia per i patch ViT
                arrow_scale=15.0   # Moltiplicatore di visibilità delle frecce
            )
            
            heatmap_view = draw_heat_map(
                normal_field, 
                raw_shape=raw_frame.shape,
                max_force=2.0 # Increase this if the screen turns red too easily
            )
            
            # 5. Combine the views horizontally
            combined_view = np.hstack((output_view, heatmap_view))
            
            # Add labels
            cv2.putText(combined_view, "Force Field", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined_view, "Normal Heatmap", (raw_frame.shape[1] + 10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 6. Show the screens
            cv2.imshow("DIGIT Raw Frame", raw_frame)
            cv2.imshow("Sparsh Multi-View", combined_view)
            
            if cv2.waitKey(1) == 27:
                break
                
    except Exception as e:
        import traceback
        print("\nERRORE CRITICO NELLA PIPELINE:")
        traceback.print_exc()
        
    finally:
        print("Disconnessione dal sensore...")
        digit_sensor.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()