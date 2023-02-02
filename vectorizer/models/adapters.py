
import torch
from torch import nn


class FeedForwardAdapter(nn.Module):
    # Feed-forward adapter with skip-connection following: https://arxiv.org/abs/1902.00751
    def __init__(self, hidden_size, adapter_dim, adapter_init_std):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim
        self.adapter_init_std = adapter_init_std

        self.adapter = nn.Sequential(
            nn.Linear(self.hidden_size, self.adapter_dim),
            nn.ReLU(),
            nn.Linear(self.adapter_dim, self.hidden_size),
        )

        self._init_weights()

    def forward(self, input_tensor):
        hidden_states = self.adapter(input_tensor)
        return hidden_states + input_tensor  # skip connection

    def _init_weights(self):
        # Initialize adapters close to identity function
        # See https://arxiv.org/abs/1902.00751, Section 3.6
        for m in self.adapter:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=self.adapter_init_std)
                m.bias.data.zero_()
