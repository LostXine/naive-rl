import torch
import torch.nn as nn


class FCN3(nn.Module):
    def __init__(self, state_len, action_len, hidden_len):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_len, hidden_len),
            nn.Linear(hidden_len, hidden_len),
            nn.Linear(hidden_len, action_len),
        )

    def forward(self, x):
        return self.layers(x)

