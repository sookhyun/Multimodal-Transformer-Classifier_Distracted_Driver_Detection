import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3, bidirectional=True):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1) # 1 output node
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the final hidden state (from last layer)
        if self.lstm.bidirectional:
            h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_final = h_n[-1]
        
        # Classification head
        out = self.classifier(h_final)
        return out

if __name__ == "__main__":
    batch_size = 16
    seq_len = 40
    num_features = 27
    num_classes = 4

    X = torch.randn(batch_size, seq_len, num_features)
    y = torch.randint(0, num_classes, (batch_size,))

    model = LSTMClassifier(input_dim=num_features, num_classes=num_classes)
    out = model(X)
    print(out.shape)  # (16, 4)

