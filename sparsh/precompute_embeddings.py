import os
import io
import gc
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from sparsh_encoder import SparshEncoder

def load_bin_image(io_buf):
    img = Image.open(io.BytesIO(io_buf))
    return np.array(img)

def precompute_facebook_dataset_low_ram(dataset_root, save_path):
    print("[PrecomputeEmbeddings] Initializing Sparsh Encoder (Optimized for 4GB RAM)...")
    encoder = SparshEncoder(device='cuda')
    saved_embeddings = []
    
    batch_dir = os.path.join(dataset_root, "sphere", "batch_1")
    
    print("[PrecomputeEmbeddings] Loading master force labels...")
    forces_file = os.path.join(batch_dir, "dataset_slip_forces.pkl")
    with open(forces_file, "rb") as f:
        labels_data = pickle.load(f)
        
    trajectories = labels_data['trajectories']
    print(f"Found {len(trajectories)} unique physical interactions.")
    
    # ==========================================
    # THE SMART 4GB RAM MANAGER
    # ==========================================
    current_chunk_id = -1
    current_images = None
    
    def fetch_image(global_idx):
        nonlocal current_chunk_id, current_images
        
        # Meta chunked them into lists of exactly 50,000
        chunk_id = global_idx // 50000
        local_idx = global_idx % 50000
        
        # If we need a frame from a new file, swap them out of RAM!
        if chunk_id != current_chunk_id:
            if current_images is not None:
                del current_images
                gc.collect() # Force clear the old 50k frames from RAM
            
            chunk_file = os.path.join(batch_dir, f"dataset_digit_{chunk_id:02d}.pkl")
            print(f"\n[RAM Manager] Swapping to new image chunk: {os.path.basename(chunk_file)}...")
            with open(chunk_file, "rb") as f:
                current_images = pickle.load(f)
            current_chunk_id = chunk_id
            
        return load_bin_image(current_images[local_idx])
        
    # ==========================================
    # THE EXTRACTION LOOP
    # ==========================================
    for traj_id, traj_data in tqdm(trajectories.items(), desc="Processing Trajectories"):
        global_indexes = traj_data['indexes']
        forces = traj_data['forces']
        
        if len(global_indexes) == 0:
            continue
            
        # 1. The Reference Frame (First frame before it touches anything)
        ref_frame = fetch_image(global_indexes[0])
        
        # --- THE ALIGNMENT FIX ---
        # Find whichever array is shorter and only loop that many times
        valid_frames = min(len(global_indexes), len(forces))
        
        # 2. Loop through the motion
        for i in range(1, valid_frames):
            active_global_idx = global_indexes[i]
        
            
            # The 6-Channel Fix: Look 5 steps back in THIS trajectory's timeline
            idx_minus_5 = max(0, i - 5) 
            past_global_idx = global_indexes[idx_minus_5]
            
            active_frame = fetch_image(active_global_idx)
            frame_minus_5 = fetch_image(past_global_idx)
            
            current_force = torch.tensor(forces[i], dtype=torch.float32)
            
            # 3. Extract embedding on the RTX 2050
            embedding = encoder.get_embeddings(ref_frame, active_frame, frame_minus_5)
            
            saved_embeddings.append({
                "embedding": embedding.squeeze(0).cpu(), 
                "force": current_force
            })
            
            # Aggressive loop cleanup
            del active_frame
            del frame_minus_5
            
        # Aggressive trajectory cleanup
        del ref_frame
        gc.collect()
        torch.cuda.empty_cache()

    print("\nSaving precomputed embeddings to disk...")
    torch.save(saved_embeddings, save_path)
    print(f"Successfully processed {len(saved_embeddings)} total frames and saved to {save_path}!")

if __name__ == "__main__":
    precompute_facebook_dataset_low_ram("../datasets/digit-force", "../datasets/digit_t1_embeddings.pt")