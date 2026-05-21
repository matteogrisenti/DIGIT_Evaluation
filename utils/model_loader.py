"""
model_loader.py
---------------
Responsible for constructing the Sparsh ForceFieldModule (encoder + decoder)
and loading the unified checkpoint weights.

Usage:
    from model_loader import load_sparsh_models
    model = load_sparsh_models(checkpoints_dir, device)
"""

import os

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from sparsh.tactile_ssl.model.vision_transformer import vit_base
from sparsh.tactile_ssl.downstream_task.forcefield_sl import ForceFieldModule


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHECKPOINT_REL_PATH = os.path.join(
    "forcefield_decoder",
    "digit_t1_forcefield_dino_vitbase_bg",
    "checkpoints",
    "last.ckpt",
)

_SPARSH_CONFIG_REL_PATH = os.path.join(
    "config", "experiment", "downstream_task", "forcefield", "digit_dino.yaml"
)

# Minimal ssl_config required by ForceFieldModule's constructor.
_DUMMY_SSL_CONFIG = {
    "img_sz": (224, 224),
    "loss": {
        "with_sl_supervision": False,
        "with_mask_supervision": False,
        "with_ssim": False,
        "min_depth": 0.1,
        "max_depth": 10.0,
    },
    "pose_estimator": {"num_encoder_layers": 18},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sparsh_models(checkpoints_dir: str, sparsh_submodule_path: str, device: torch.device) -> ForceFieldModule:
    """
    Build the ForceFieldModule (encoder + decoder) and load unified weights.

    Args:
        checkpoints_dir:       Root directory that contains the checkpoint tree.
        sparsh_submodule_path: Absolute path to the sparsh git submodule root,
                               used to locate the Hydra YAML config.
        device:                Torch device to move the model onto.

    Returns:
        A fully-loaded ForceFieldModule in eval mode.
    """
    print(f"Loading models onto device: {device}")

    encoder = _build_encoder()
    decoder = _build_decoder(sparsh_submodule_path)
    model_module = _wrap_in_module(encoder, decoder)
    _load_weights(model_module, checkpoints_dir, device)

    model_module.to(device)
    model_module.eval()

    print(f"ForceFieldModule ready on {device}")
    return model_module


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_encoder():
    """Instantiate the DINOv2 ViT-Base encoder with 6-channel input."""
    print("Instantiating encoder (ViT-Base, 6-channel)...")
    return vit_base(
        img_size=224,
        in_chans=6,            # I_t (3ch) concatenated with I_{t-5} (3ch)
        patch_size=16,
        num_register_tokens=1,
    )


def _build_decoder(sparsh_submodule_path: str):
    """Instantiate the DPT Force-Field decoder via Hydra config."""
    print("Instantiating decoder (DPT Force Field) from Hydra config...")
    config_path = os.path.join(sparsh_submodule_path, _SPARSH_CONFIG_REL_PATH)
    cfg = OmegaConf.load(config_path)
    return instantiate(cfg.task.model_task)


def _wrap_in_module(encoder, decoder) -> ForceFieldModule:
    """Combine encoder and decoder inside ForceFieldModule."""
    print("Wrapping encoder + decoder into ForceFieldModule...")
    return ForceFieldModule(
        model_encoder=encoder,
        model_task=decoder,
        optim_cfg=None,
        scheduler_cfg=None,
        ssl_config=_DUMMY_SSL_CONFIG,
    )


def _load_weights(model_module: ForceFieldModule, checkpoints_dir: str, device: torch.device) -> None:
    """Load unified checkpoint weights (strict=False to skip unused heads)."""
    checkpoint_path = os.path.join(checkpoints_dir, _CHECKPOINT_REL_PATH)
    print(f"Loading weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Support both Lightning-style and plain state-dict checkpoints.
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # strict=False: PoseEstimator head is present in the checkpoint but unused.
    model_module.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully.")