"""
sparsh_force_field_test.py
-------
Entry point for the Sparsh DIGIT tactile-sensing pipeline.

Responsibilities here are intentionally minimal:
  - Bootstrap path / hardware compatibility shims
  - Wire together the loader, sensor, and display modules
  - Run the frame loop

Everything else lives in its own module.
"""

# ==============================================================================
# Bootstrap: xformers shim must be installed before any DINOv2 import.
# ==============================================================================
from utils.xformers_mock import patch_xformers
patch_xformers()
# ==============================================================================

import os
import sys
import time
import collections
import traceback
import multiprocessing as mp

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: make the sparsh submodule importable
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SPARSH_SUBMODULE_PATH = os.path.join(CURRENT_DIR, "sparsh")
if SPARSH_SUBMODULE_PATH not in sys.path:
    sys.path.insert(0, SPARSH_SUBMODULE_PATH)

# ---------------------------------------------------------------------------
# Project imports (after path is configured)
# ---------------------------------------------------------------------------
from digit_interface.digit_windows import Digit
from utils.model_loader import load_sparsh_models
from utils.display_tools import build_combined_view, show_frames, should_quit, teardown_windows
from utils.dataset_write import force_dataset_worker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = os.path.join(CURRENT_DIR, "outputs_sparsh", "checkpoints")
WARMUP_FRAMES = 90          # Frames discarded before capturing background
TEMPORAL_STRIDE = 5         # I_t vs I_{t-5}: buffer depth = stride + 1
BUFFER_SIZE = TEMPORAL_STRIDE + 1

RECORD_DATASET = True                # Activate/Deactivate dataset recording
RECORDING_DURATION_SEC = 90          # Duration of the recording in seconds
DATASET_OUTPUT_DIR = os.path.join(CURRENT_DIR, "force_datasets")

# ---------------------------------------------------------------------------
# Frame pre-processing
# ---------------------------------------------------------------------------

def pre_process_frame(frame: np.ndarray, bg_frame: np.ndarray) -> np.ndarray:
    """
    Prepare a raw sensor frame for model inference.

    Steps:
      1. Background subtraction (improves cross-sensor generalisation).
      2. Cast to float32 and normalise to [0, 1].
      3. Resize to 224×224 (ViT native resolution).
    """
    diff = cv2.subtract(frame, bg_frame)
    normalised = diff.astype(np.float32) / 255.0
    return cv2.resize(normalised, (224, 224))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, input_pair: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the ForceFieldModule on a (I_t ‖ I_{t-5}) 6-channel input.

    Args:
        model:      Loaded ForceFieldModule in eval mode.
        input_pair: (224, 224, 6) float32 array.
        device:     Torch device.

    Returns:
        (normal_field, shear_field) as (H, W, C) numpy arrays.
    """
    tensor = (
        torch.from_numpy(input_pair)
        .permute(2, 0, 1)   # HWC → CHW
        .unsqueeze(0)        # add batch dim
        .to(device)
    )

    with torch.no_grad():
        pred = model(tensor)

    def to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.squeeze(0).permute(1, 2, 0).cpu().numpy()

    return to_numpy(pred["normal"]), to_numpy(pred["shear"])


# ---------------------------------------------------------------------------
# Sensor helpers
# ---------------------------------------------------------------------------

def capture_background(sensor: Digit, warmup: int = WARMUP_FRAMES) -> np.ndarray:
    """Discard `warmup` frames, then capture a clean background."""
    print(f"Warming up sensor ({warmup} frames)… do not touch the gel.")
    for _ in range(warmup):
        sensor.get_frame()
    bg = sensor.get_frame()
    print("Background captured.")
    return bg


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load model
    model = load_sparsh_models(CHECKPOINTS_DIR, SPARSH_SUBMODULE_PATH, device)

    # 2. Connect sensor
    print("Connecting to DIGIT sensor…")
    sensor = Digit(device_index=0, name="Windows_DIGIT")
    sensor.connect()

    bg_frame = capture_background(sensor)

    # 3. Temporal frame buffer (size = stride + 1, so index 0 is I_{t-5})
    frame_history: collections.deque = collections.deque(maxlen=BUFFER_SIZE)
    last_frame: np.ndarray | None = None

    # 4. Dataset recording subprocess setup
    is_recording = RECORD_DATASET
    dataset_queue = None
    worker_process = None

    if is_recording:
        # Create a multiprocessing queue and start the dataset writer worker process
        dataset_queue = mp.Queue(maxsize=1000)
        worker_process = mp.Process(
            target=force_dataset_worker, 
            args=(dataset_queue, DATASET_OUTPUT_DIR, RECORDING_DURATION_SEC)
        )
        worker_process.start()
        print(f"Main loop (PID: {os.getpid()}) sta inviando i dati al Worker.")

    print("\nPipeline running. Press ESC to quit.")

    try:
        while True:
            raw_frame = sensor.get_frame()
            # print(f"DEBUG: Frame dimensions: {raw_frame.shape}")
            processed = pre_process_frame(raw_frame, bg_frame)

            t_attuale = time.time()

            # Skip duplicate frames (sensor may repeat the last frame)
            if last_frame is not None and np.array_equal(processed, last_frame):
                cv2.waitKey(1)
                continue

            frame_history.append(processed)
            last_frame = processed

            # Wait until the buffer is full
            if len(frame_history) < BUFFER_SIZE:
                continue

            # 4. Build 6-channel temporal input
            I_t = frame_history[-1]
            I_t5 = frame_history[0]
            input_pair = np.concatenate([I_t, I_t5], axis=-1)  # (224, 224, 6)

            # 5. Inference
            normal_field, shear_field = run_inference(model, input_pair, device)

            # 6. Send data to the dataset worker if recording and worker is alive
            if is_recording and worker_process.is_alive():
                try:
                    # Send a tuple of (normal_field, shear_field, timestamp) to the worker
                    dataset_queue.put_nowait((normal_field, shear_field, t_attuale))
                except mp.queues.Full:
                    pass # If the queue is full, we skip this frame to avoid blocking the main loop
            elif is_recording and not worker_process.is_alive():
                is_recording = False # Stop trying to send data if the worker has died

            # 7. Render and display
            combined = build_combined_view(normal_field, shear_field, raw_frame)
            show_frames(raw_frame, combined)

            if should_quit():
                break

    except Exception:
        print("\nCRITICAL PIPELINE ERROR:")
        traceback.print_exc()

    finally:
        print("Disconnecting from sensor…")
        sensor.disconnect()
        teardown_windows()
        print("Sensor disconnected")

        if worker_process is not None:
            if worker_process.is_alive():
                print("\n[Main] Il worker è ancora in esecuzione. Inviando segnale di arresto...")
                try:
                    dataset_queue.put(None)  # Segnale di stop
                except Exception:
                    pass
                worker_process.join(timeout=5)
                
                # Se dopo 5 secondi è ancora vivo, forziamo la chiusura
                if worker_process.is_alive():
                    print("[Main] Il worker non risponde. Chiusura forzata.")
                    worker_process.terminate()
            else:
                print("\n[Main] Il worker ha concluso naturalmente il suo lavoro.")
                # Dobbiamo comunque chiamare join per liberare le risorse del sistema operativo!
                worker_process.join() 
            
            # --- FIX ANTI-DEADLOCK PER LA CODA ---
            if dataset_queue is not None:
                # 1. Svuotiamo fisicamente la coda da eventuali frame rimasti intrappolati
                while not dataset_queue.empty():
                    try:
                        dataset_queue.get_nowait()
                    except Exception:
                        break
                
                # 2. Diciamo a Python di NON aspettare che i dati vengano consumati (risolve il freeze!)
                dataset_queue.cancel_join_thread()
                dataset_queue.close()
                
            print("[Main] Sottoprocessi chiusi correttamente.")


if __name__ == "__main__":
    main()