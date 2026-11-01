"""
models/sensor_har.py
--------------------
PyTorch model class definitions for the Sensor HAR model.
These definitions MUST exactly match the architecture used during training
(har_bilstm_sensor_enhanced_colab.py) so that state_dict keys align correctly
when loading from the checkpoint.
"""

import torch
from torch import nn


class HARBiLSTM(nn.Module):
    """
    Reference model: single-layer Bidirectional LSTM.

    Input : (batch, 128, 9)  — 128 time steps, 9 sensor channels
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 64,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        # Concatenate the last forward and backward hidden states
        final_states = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(final_states)


class HARConvBiLSTM(nn.Module):
    """
    Temporal CNN + BiLSTM with attention-weighted time pooling.
    This is the model selected by the training notebook (96.74 % val accuracy).

    Input : (batch, 128, 9)  — 128 time steps, 9 sensor channels
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 64,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN expects (batch, channels, time) — transpose in/out
        x = self.temporal_cnn(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(x)
        # Attention pooling over time steps
        weights = torch.softmax(
            self.attention(sequence).squeeze(-1), dim=1
        )
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(pooled))


# Map model name string (stored in checkpoint) → class
SENSOR_MODEL_REGISTRY: dict[str, type] = {
    "Original BiLSTM": HARBiLSTM,
    "Temporal CNN-BiLSTM": HARConvBiLSTM,
}

