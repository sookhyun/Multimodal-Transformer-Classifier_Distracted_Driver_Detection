

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super(ResNetLayer, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out



class ResNetClassifier(nn.Module):
    def __init__(self, input_channels=27):
        super(ResNetClassifier, self).__init__()

        # ResNet layers
        self.layer1 = ResNetLayer(input_channels, 2*input_channels, kernel_size=8)
        self.layer2 = ResNetLayer(2*input_channels, 4*input_channels, kernel_size=5)
        self.layer3 = ResNetLayer(4*input_channels, 8*input_channels, kernel_size=3)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected classifier
        self.fc = nn.Linear(8*input_channels, 1)

    def forward(self, x):
        # input x shape: (batch_size, seq_len, num_features)
        x = x.transpose(1, 2)
        # x shape: (batch_size, num_features, seq_len)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x).squeeze(-1)  # -> (batch_size, 256)
        x = self.fc(x)
        return x





# Example: forward pass
if __name__ == "__main__":
    input_channels = 27
    seq_len = 40
    batch_size = 16
    num_classes = 2
    
    # Create random multivariate time series
    X = torch.randn(batch_size, input_channels, seq_len)  # (batch_size, channels, seq_len)
    y = torch.randint(0, num_classes, (batch_size,))      # Random labels

    block = ResNetLayer(in_channels=27, out_channels=64, kernel_size=7)

    # Forward pass for layer
    x = torch.randn(8, 27, 40)  # batch_size=8, channels=27, seq_len=40
    out = block(x)
    print(out.shape)  # -> torch.Size([8, 64, 40])

    # Initialize model
    model = ResNetClassifier(input_channels=input_channels)

    # Forward pass
    logits = model(X)

    # Check output shape
    print("Output shape:", logits.shape)
    assert logits.shape == (batch_size, 1), "Output shape is incorrect!"

    # Compute loss
    loss = nn.BCEWithLogitsLoss()(logits.squeeze(), y.float())
    print("Loss:", loss.item())

    # Backward pass
    loss.backward()
    print("Backward pass successful!")




