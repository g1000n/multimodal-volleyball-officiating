"""
model.py

CNN-LSTM hybrid for classifying referee gesture sequences from
normalized, resampled MediaPipe keypoint sequences.

Input shape per clip: (sequence_length, num_features)
  - sequence_length: fixed number of frames after resampling (e.g. 50)
  - num_features: landmarks * 3 (x, y, visibility) — 24 for the 8
    upper-body landmarks used in extract_keypoints.py

The 1D conv layer first picks up short-range local motion patterns
frame-to-frame, then the LSTM models the longer temporal sequence
(the overall arc of the gesture: raise -> point -> drop).
"""

import torch
import torch.nn as nn


class GestureCNNLSTM(nn.Module):
    def __init__(self, input_size, num_classes, cnn_channels=64,
                 lstm_hidden=128, lstm_layers=1, dropout=0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)          # -> (batch, input_size, seq_len) for Conv1d
        x = self.relu(self.conv1(x))
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)          # -> back to (batch, seq_len, cnn_channels)

        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]           # final layer's hidden state: (batch, lstm_hidden)
        last_hidden = self.dropout2(last_hidden)

        logits = self.fc(last_hidden)   # (batch, num_classes)
        return logits