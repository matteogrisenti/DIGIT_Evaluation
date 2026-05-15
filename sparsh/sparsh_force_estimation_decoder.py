import torch.nn as nn

class SparshT1ForceDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        
        # The Learned Query
        self.learned_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Cross-Attention Module
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # 2-Layer MLP (768 -> 192 -> 3)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 192),
            nn.GELU(),
            nn.Linear(192, 3) 
        )

    def forward(self, embeddings):
        """
        Takes the latent embeddings from the SparshEncoder and outputs Fx, Fy, Fz.
        """
        batch_size = embeddings.shape[0]
        
        # Apply Cross-Attention
        query = self.learned_query.expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attention(query=query, key=embeddings, value=embeddings)
        
        # Pool and pass through MLP
        pooled_features = self.layer_norm(attn_out).squeeze(1)
        forces_3d = self.mlp(pooled_features)
        
        return forces_3d