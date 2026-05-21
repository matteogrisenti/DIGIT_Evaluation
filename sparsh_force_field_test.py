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
import collections
import traceback

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = os.path.join(CURRENT_DIR, "outputs_sparsh", "checkpoints")
WARMUP_FRAMES = 10          # Frames discarded before capturing background
TEMPORAL_STRIDE = 5         # I_t vs I_{t-5}: buffer depth = stride + 1
BUFFER_SIZE = TEMPORAL_STRIDE + 1


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

    print("\nPipeline running. Press ESC to quit.")

    try:
        while True:
            raw_frame = sensor.get_frame()
            processed = pre_process_frame(raw_frame, bg_frame)

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

            # 6. Render and display
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


if __name__ == "__main__":
    main()