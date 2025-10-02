"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn

class TinyVGG(nn.Module):
    """Creates the TinyVGG model.

    See the original model architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
        input_shape: Number of input channels.
        hidden_units: Number of hidden units in layers.
        output_shape: Number of output units.
    """
    def __init__(
            self,
            input_shape: int,
            hidden_units: int,
            output_shape: int
    ) -> None:
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.

        Args:
            x: Input tensor of shape (N, C, H, W), where
                - N is the batch size,
                - C is the number of channels,
                - H is the height,
                - W is the width.
        
        Returns:
            Tensor of data that pass through
            `block_1`, `block_2`, and `classifier`.
        """
        return self.classifier(self.block_2(self.block_1(x)))