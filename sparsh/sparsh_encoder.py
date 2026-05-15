import torch
import timm
import numpy as np
from torchvision import transforms
from huggingface_hub import hf_hub_download, list_repo_files

class SparshEncoder:
    def __init__(self, device='cuda'):
        self.device = device
        
        # 1. We switch back to Patch 16 because the downloaded weights demand it!
        print("[SparshEncoder] Loading Custom ViT Architecture (Patch 16, 6-channel, 4 Registers)...")
        self.backbone = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=False, 
            in_chans=6, 
            num_classes=0,
            class_token=False,  
            reg_tokens=1,       
            init_values=1e-5,
            global_pool='avg'   # <--- Add this line here!
        )
        
        repo_id = "facebook/sparsh-dino-base"
        
        # 2. Find and load the safetensors file
        available_files = list_repo_files(repo_id)
        weight_filename = None
        for f in ["model.safetensors", "pytorch_model.bin"]:
            if f in available_files:
                weight_filename = f
                break
        if not weight_filename:
            for f in available_files:
                if f.endswith(('.pth', '.pt', '.safetensors')):
                    weight_filename = f
                    break

        print(f"[SparshEncoder] Loading weights from {weight_filename} cache...")
        model_path = hf_hub_download(repo_id=repo_id, filename=weight_filename)
        
        if weight_filename.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
            
        # ---------------------------------------------------------
        # 3. THE TRANSLATION LAYER: Fix Meta's internal naming
        # ---------------------------------------------------------
        print("[SparshEncoder] Translating Meta's internal weight names to match timm...")
        
        # Rename plural to singular
        if "register_tokens" in state_dict:
            state_dict["reg_token"] = state_dict.pop("register_tokens")
            
        # Delete custom frequency bands (timm handles positional math automatically)
        if "pos_embed.frequency_bands" in state_dict:
            del state_dict["pos_embed.frequency_bands"]
            
        # 4. INJECT WEIGHTS (strict=False ignores any missing generic positional embeddings)
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.to(self.device)
        self.backbone.eval()
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_frame(self, ref_frame, frame):
        diff = frame.astype(np.float32) - ref_frame.astype(np.float32)
        diff = np.clip(diff + 128, 0, 255).astype(np.uint8)
        return self.transform(diff)

    def get_embeddings(self, ref_frame, frame_t, frame_t_minus_5):
        tensor_t = self.preprocess_frame(ref_frame, frame_t)
        tensor_t_minus_5 = self.preprocess_frame(ref_frame, frame_t_minus_5)
        
        input_tensor = torch.cat([tensor_t, tensor_t_minus_5], dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embeddings = self.backbone.forward_features(input_tensor)
            
        return embeddings