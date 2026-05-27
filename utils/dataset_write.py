import multiprocessing as mp
import time
import os
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Force Dataset Worker (Processo Separato)
# ---------------------------------------------------------------------------
def force_dataset_worker(queue: mp.Queue, output_dir: str, duration: float) -> None:
    """
    Worker che accumula i campi di forza in uscita dal modello e li salva su disco.
    """
    normal_frames = []
    shear_frames = []
    timestamps = []
    start_time = None
    
    print(f"\n[Worker] Registrazione forze avviata (PID: {os.getpid()})")
    
    while True:
        item = queue.get()
        if item is None:  # Segnale di arresto
            break
            
        norm, shear, t = item
        
        if start_time is None:
            start_time = t
            
        normal_frames.append(norm)
        shear_frames.append(shear)
        timestamps.append(t)
        
        if (t - start_time) > duration:
            break

    # -- Salvataggio --
    if normal_frames:
        print(f"\n[Worker] Tempo scaduto. Salvataggio di {len(normal_frames)} misurazioni di forza...")
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"force_record_{timestamp_str}.npz")
        
        # Salviamo in float32 per mantenere la precisione dell'inferenza
        np.savez_compressed(
            output_file,
            normal=np.array(normal_frames, dtype=np.float32),
            shear=np.array(shear_frames, dtype=np.float32),
            timestamps=np.array(timestamps, dtype=np.float64)
        )
        print(f"[Worker] Dataset forze salvato: {output_file}")
    
    # Pulizia memoria
    normal_frames.clear()
    shear_frames.clear()
    timestamps.clear()



# ---------------------------------------------------------------------------
# Raw Frame Image Worker (Processo Separato)
# ---------------------------------------------------------------------------
def raw_frame_worker(queue: mp.Queue, output_dir: str, duration: float) -> None:
    """
    Worker che riceve i raw frame dal sensore e li salva come singole immagini PNG
    in una cartella dedicata.
    """
    print(f"\n[Raw Worker] Salvataggio frame singoli avviato (PID: {os.getpid()})")
    
    start_time = None
    frames_saved = 0
    frames_dir = ""
    
    while True:
        item = queue.get()
        if item is None:  # Segnale di arresto
            break
            
        frame, t = item
        
        # Inizializzazione al primo frame ricevuto
        if start_time is None:
            start_time = t
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            
            # Crea una sottocartella dedicata per i frame di questa sessione
            frames_dir = os.path.join(output_dir, f"raw_frames_{timestamp_str}")
            os.makedirs(frames_dir, exist_ok=True)
            print(f"[Raw Worker] Cartella creata: {frames_dir}")
            
        # Salvataggio del frame come immagine 
        # Usiamo PNG (compressione lossless) per mantenere i dati intatti
        frame_filename = os.path.join(frames_dir, f"frame_{frames_saved:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frames_saved += 1
        
        # Controllo durata
        if (t - start_time) > duration:
            break

    print(f"\n[Raw Worker] Chiusura. Salvati {frames_saved} frame nella cartella: {frames_dir}")