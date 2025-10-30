
from Embedding import *
from SelfAttentionEncoder import *

class TransformerTimeseriesClassifier(nn.Module):
    def __init__(self, params: Params): #num_feature, dim_embed, num_heads, num_layers, num_classes, dropout=0.1):
        super(TransformerTimeseriesClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = TemporalFeatureEmbedding(params.num_features, params.seq_len, params.dim_embed)
        #nn.Linear(params.num_features, params.dim_embed)
        
        # Transformer Encoder layers
        self.encoder_layer = SelfAttentionEncoderLayer(params 
                        #nn.TransformerEncoderLayer(

        )
        self.encoder = SelfAttentionEncoder(self.encoder_layer, num_layers=params.num_layers) 
        #nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling (mean or [CLS]-style)
        self.pool = nn.AdaptiveAvgPool1d(1) 
        #self.norm = nn.LayerNorm(dim_embed) # for  CLS pooling
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(params.dim_embed, params.dim_embed // 2),
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.dim_embed // 2, 1) # 1 -> num_features for multi-class
        )
        

    def forward(self, x):
        """
        x: (batch_size, seq_len, num_features)
        """
        # Embed input
       
        x = self.embedding(x)                    # [B, T*F, dim_embed]
       
        # Encode with self-attention
        encoded = self.encoder(x)
      
        # Pooling: 
        # 1. Take simple Mean 
        #pooled = encoded.mean(dim=1)
        
        # 2. global average pooling over 1D tokens
        encoded = encoded.transpose(1, 2)        # [B, dim_embed, T*F]
        pooled = self.pool(encoded).squeeze(-1)  # [B, dim_embed]

        # 3. Take CLS token
           # Prepend CLS token
        #cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        #x = torch.cat([cls_tokens, x], dim=1)      # [B, 1 + T*F, d_model]        
           # Take CLS representation (first token)
        #pooled = self.norm(encoded[:, 0, :])         # [B, dim_embed]
        
        # Classify      
        logits = self.classifier(pooled)         # [B, num_classes]

        return logits





# Example usage
if __name__ == "__main__":
    batch_size = 16
    seq_len = 40
    num_feature = 30     # e.g., 30 sensor features
    dim_embed = 10
    num_heads = 4
    num_layers = 2
    num_classes = 2

    model = TransformerTimeseriesClassifier(params)
    dummy_input = torch.randn(batch_size, seq_len, num_feature)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    logits = model(dummy_input)
    print("Output shape:", logits.shape)  # (batch_size, num_classes)
    loss = nn.BCEWithLogitsLoss()(logits.squeeze(), labels.float())
    print("Loss:", loss.item())