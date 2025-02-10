import torch
import torch.nn as nn
import torchvision.models as models

class CNN_LSTM(nn.Module):
    def __init__(self, seq_length=30, lstm_hidden_size=256, num_classes=2):
        super(CNN_LSTM, self).__init__()

        # Load pre-trained ResNet18
        self.resnet = models.resnet18(pretrained=True)

        # Remove classification ResNet layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.resnet_out_features = 512  # Feature dimension from ResNet

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.resnet_out_features,  # From CNN
            hidden_size=lstm_hidden_size,  # Hidden state
            num_layers=2,
            batch_first=True  # Batch size as first dimension
        )

        # Final classificator
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        cnn_features = []

        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            out = self.resnet(frame)
            out = out.view(batch_size, -1)  # Flattening
            cnn_features.append(out) 

        cnn_features = torch.stack(cnn_features, dim=1)  # (B, T, Feature_Size)

        lstm_out, _ = self.lstm(cnn_features)  # (B, T, LSTM_Hidden_Size)

        final_out = lstm_out[:, -1, :]  # (B, LSTM_Hidden_Size)
        return self.fc(final_out)  # Final classification

