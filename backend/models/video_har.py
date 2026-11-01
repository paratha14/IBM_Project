"""
models/video_har.py
-------------------
PyTorch model class definitions for the Video (Pose) HAR model.
This definition MUST exactly match the architecture in
penn_action_pose_bilstm_optimized_final.py so that the state_dict loads cleanly.
"""

import torch
from torch import nn


class PoseMotionBiLSTM(nn.Module):
    """
    Two-layer Bidirectional LSTM that classifies human actions from
    compact pose-feature sequences.

    Input : (batch, 64, 91)  — 64 sampled frames × 91 pose features per frame
    Output: (batch, num_classes)

    The 91 features per frame are:
      - 26 normalized joint coordinates  (13 joints × x + y)
      - 13 joint visibility scores
      - 26 frame-to-frame coordinate velocities
      - 26 parent-relative bone vectors   (13 joints × dx + dy)
    """

    def __init__(
        self,
        input_size: int = 91,
        hidden_size: int = 64,
        num_classes: int = 15,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        # Concatenate last forward + backward hidden states
        final_state = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(final_state)

