
from MultiHeadSelfAttention import *
import copy

class FeedForward(nn.Module):

    def __init__(self, dim_embed, dim_ff, dropout=0.1):
        super(FeedForward, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim_embed, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x) 



class ResidualConnection(nn.Module):

    def __init__(self, dim_embed):
        super(ResidualConnection, self).__init__()
        self.normalized = nn.LayerNorm(dim_embed)


    def forward(self, x, sublayer):      
        
        return self.normalized(x + sublayer(x))



class SelfAttentionEncoderLayer(nn.Module):
    "Encoder is made up of two sublayers, self-attn and feed forward. "
    def __init__(self, params: Params):

        super(SelfAttentionEncoderLayer, self).__init__()
        self.dim_embed = params.dim_embed
        self.self_attn = MultiHeadSelfAttention(params)
        self.feed_forward = FeedForward(params.dim_embed, params.dim_ff, dropout=params.dropout)
        self.sublayers = nn.ModuleList([copy.deepcopy(ResidualConnection(params.dim_embed)) for _ in range(2)])

    def forward(self, x, mask=None):
        # x -> self attention -> residual connection -> feed forward -> residual connection -> out
        x = self.sublayers[0](x, lambda x: self.self_attn(x, mask)[0])
        return self.sublayers[1](x, self.feed_forward)
        


class SelfAttentionEncoder(nn.Module):
    
    def __init__(self, layer :SelfAttentionEncoderLayer, num_layers):
        super(SelfAttentionEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.normalized = nn.LayerNorm(layer.dim_embed) # NormalizeLayer(layer.dim_embed)
        for param in self.layers.parameters():
            param.requires_grad = True
        
    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.normalized(x)

