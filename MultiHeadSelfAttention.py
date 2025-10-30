import torch
import torch.nn as nn
import torch.nn.functional as F

from parameters import Params

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, params: Params): #dim_embed=128, num_heads=4, dropout=0.1):
        super().__init__()
        assert params.dim_embed % params.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.dim_embed = params.dim_embed
        self.num_heads = params.num_heads
        self.dim_head = params.dim_embed // params.num_heads  # 32 if 128/4

        # Linear layers for Query, Key, and Value
        self.W_Q = nn.Linear(self.dim_embed, self.dim_embed)
        self.W_K = nn.Linear(self.dim_embed, self.dim_embed)
        self.W_V = nn.Linear(self.dim_embed, self.dim_embed)

        # Final linear projection after concatenating heads
        self.W_O = nn.Linear(self.dim_embed, self.dim_embed)
        self.dropout = nn.Dropout(p=params.dropout)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, dim_embed)
        """
        B, T, E = x.shape
        H, D = self.num_heads, self.dim_head

        # Project into Query, Key and Value
        Q = self.W_Q(x)  # (B, T, E)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape for multi-head attention (B, H, T, D)
        Q = Q.view(B, T, H, D).transpose(1, 2)
        K = K.view(B, T, H, D).transpose(1, 2)
        V = V.view(B, T, H, D).transpose(1, 2)

        # Compute attention scores and attention weights
        attn_scores = torch.matmul(Q,K.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, T, D)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T, D)

        # Concatenate heads back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, E)

        # Final linear projection
        out = self.W_O(attn_output)  # (B, T, E)

        return out, attn_weights  # returning attention weights for visualization

# Example: forward pass
if __name__ == "__main__":
    params = Params()
    B, T, E = 16, 40, 128
    num_heads = 4
    x = torch.randn(B, T, E)

    attn = MultiHeadSelfAttention(params)
    out, weights = attn(x)

    print("Input:", x.shape)
    print("Attention weights:", weights.shape)
    print("Output:", out.shape)