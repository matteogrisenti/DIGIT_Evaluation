import multiprocessing as mp
import time
import os
import numpy as np

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