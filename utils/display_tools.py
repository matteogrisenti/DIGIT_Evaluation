"""
display_tools.py
----------------
Rendering helpers that turn raw model outputs into OpenCV-ready images.

Usage:
    from display_tools import build_combined_view, show_frames, WINDOW_RAW, WINDOW_MULTI
"""

import cv2
import numpy as np

from utils.draw_force_field_utility import draw_force_field
from utils.draw_heat_map_utility import draw_heat_map


# ---------------------------------------------------------------------------
# Window identifiers (single source of truth)
# ---------------------------------------------------------------------------

WINDOW_RAW = "DIGIT Raw Frame"
WINDOW_MULTI = "Sparsh Multi-View"


# ---------------------------------------------------------------------------
# Rendering parameters (tune here, not scattered through pipeline code)
# ---------------------------------------------------------------------------

_FORCE_BIAS: float = 0.0   # Set >0 to add a constant offset to all arrows
_STRIDE: int = 14           # Grid stride — matches ViT patch size
_ARROW_SCALE: float = 15.0  # Arrow length multiplier
_MAX_NORMAL_FORCE: float = 2.0  # Heatmap saturation threshold


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_combined_view(
    normal_field: np.ndarray,
    shear_field: np.ndarray,
    raw_frame: np.ndarray,
) -> np.ndarray:
    """
    Compose the side-by-side Force Field + Normal Heatmap view.

    Args:
        normal_field: (H, W, C) float array from model output.
        shear_field:  (H, W, C) float array from model output.
        raw_frame:    Original sensor frame (used to determine output shape).

    Returns:
        A single BGR image ready for cv2.imshow().
    """
    raw_shape = raw_frame.shape

    force_view = draw_force_field(
        normal_field,
        shear_field,
        raw_shape=raw_shape,
        force_bias=_FORCE_BIAS,
        stride=_STRIDE,
        arrow_scale=_ARROW_SCALE,
    )

    heat_view = draw_heat_map(
        normal_field,
        raw_shape=raw_shape,
        max_force=_MAX_NORMAL_FORCE,
        bg_threshold=0.12,  
    )

    combined = np.hstack((force_view, heat_view))
    _overlay_labels(combined, width=raw_shape[1])
    return combined


def show_frames(raw_frame: np.ndarray, combined_view: np.ndarray) -> None:
    """Push both windows to the screen."""
    cv2.imshow(WINDOW_RAW, raw_frame)
    cv2.imshow(WINDOW_MULTI, combined_view)


def should_quit() -> bool:
    """Return True when the user presses ESC."""
    return cv2.waitKey(1) == 27


def teardown_windows() -> None:
    """Destroy all OpenCV windows cleanly."""
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _overlay_labels(image: np.ndarray, width: int) -> None:
    """Burn column labels into the combined view in-place."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Force Field",    (10, 20),          font, 0.6, (255, 255, 255), 2)
    cv2.putText(image, "Normal Heatmap", (width + 75 + 10, 20),  font, 0.6, (255, 255, 255), 2)