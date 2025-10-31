import torch
import torch.nn as nn

class FullFeatureTemporalEmbedding(nn.Module):
    def __init__(self, num_features, seq_len, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.dim_embed = dim_embed

        # Project numeric feature values to dim_embed
        self.value_proj = nn.Linear(1, dim_embed)

        # Learnable time embeddings (like positional encoding)
        self.time_embed = nn.Embedding(seq_len, dim_embed)

        # Learnable feature embeddings (to identify each feature)
        self.feature_embed = nn.Embedding(num_features, dim_embed)

    
    def forward(self, x):
        # x: [B, T, F]
        B, T, F = x.shape
        assert T == self.seq_len and F == self.num_features, (
            f"Expected ({self.seq_len}, {self.num_features}), got ({T}, {F})"
        )

        # Expand scalar values -> project
        x = x.unsqueeze(-1)                            # [B, T, F, 1]
        value_emb = self.value_proj(x)                 # [B, T, F, dim_embed]

        # Time embeddings
        time_ids = torch.arange(T, device=x.device)
        time_emb = self.time_embed(time_ids)           # [T, dim_embed]
        time_emb = time_emb.unsqueeze(1).expand(T, F, self.dim_embed)
        time_emb = time_emb.unsqueeze(0).expand(B, -1, -1, -1)

        # Feature embeddings
        feat_ids = torch.arange(F, device=x.device)
        feat_emb = self.feature_embed(feat_ids)        # [F, dim_embed]
        feat_emb = feat_emb.unsqueeze(0).unsqueeze(0).expand(B, T, F, self.dim_embed)

        # Combine all embeddings
        emb = value_emb + feat_emb + time_emb          # [B, T, F, dim_embed]
        emb = emb.view(B, T * F, self.dim_embed)       # flatten to [B, T*F, dim_embed]
        return emb



class FeatureLinearProjectionTemporalEmbedding(nn.Module):
    def __init__(self, num_features, seq_len, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.dim_embed = dim_embed

        # Project numeric feature values to dim_embed
        self.feature_embed = nn.Linear(num_features, dim_embed)

        # Learnable time embeddings (like positional encoding)
        self.time_embed = nn.Embedding(seq_len, dim_embed)


    def forward(self, x):
        # x: [B, T, F]
        B, T, F = x.shape
        assert T == self.seq_len and F == self.num_features, (
            f"Expected ({self.seq_len}, {self.num_features}), got ({T}, {F})"
        )

        # Feature embeddings
        feat_emb = self.feature_embed(x)               # [B, T, dim_embed]

        # Time embeddings
        time_ids = torch.arange(T, device=x.device)
        time_emb = self.time_embed(time_ids)           # [T, dim_embed]
        time_emb = time_emb.unsqueeze(0).expand(B, -1, -1) 

        # Combine all embeddings
        emb = feat_emb + time_emb                      # [B, T, dim_embed]
        return emb



if __name__ == "__main__":
    batch_size = 2
    seq_len = 40
    num_features = 30
    d_model = 24

    # Create random dummy data
    x = torch.randn(batch_size, seq_len, num_features)

    # Initialize embedding module
    embedder = FullFeatureTemporalEmbedding(num_features, seq_len, d_model)

    # Forward pass
    out = embedder(x)

    # Check shapes and print results
    print("Input shape:", x.shape)
    
    print("Output shape for 'FullFeatureTemporalEmbedding':", out.shape)   # Expected: [B, T*F, dim_embed]
    print("Example token embedding:", out[0, 0, :5])  # first token, first 5 dims

    embedder = FeatureLinearProjectionTemporalEmbedding(num_features, seq_len, d_model)
    out = embedder(x)
    print("Output shape for 'FeatureLinearProjectionTemporalEmbedding':", out.shape)   # Expected: [B, T*F, dim_embed]    

