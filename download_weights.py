import os
from huggingface_hub import snapshot_download

# Define where we waant to save the checkpoints. 
# This directory will be given to Hydra as the default path for loading the models.
checkpoints_dir = "outputs_sparsh/checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

print("Start Force Field Decoder download (Force Field)...")
snapshot_download(
    repo_id="facebook/sparsh-digit-forcefield-decoder",
    local_dir=f"{checkpoints_dir}/forcefield_decoder",
    local_dir_use_symlinks=False  
)
print("Force Field Decoder downloaded successfully!\n")

print("Start DINO Encoder download (Sparsh DINO)...")
snapshot_download(
    repo_id="facebook/sparsh-dino-base",
    local_dir=f"{checkpoints_dir}/encoder_dino",
    local_dir_use_symlinks=False
)
print("DINO Encoder downloaded successfully!")


print(f"All files are ready in the folder: {os.path.abspath(checkpoints_dir)}")