import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SugarGridCNN(BaseFeaturesExtractor):
    """Lightweight CNN for SugarFight grid observations.

    Expects observation shape = (C, H, W) = (num_layers, Map.HEIGHT, Map.WIDTH)
    Values are already in meaningful ranges (-1,0,1 or 0~1). We do NOT normalize
    to keep semantic distances. Output feature dim = 256.
    """

    def __init__(self, observation_space: spaces.Box):
        n_channels, h, w = observation_space.shape
        super().__init__(observation_space, features_dim=256)

        # Simple stack: 3 conv blocks + global avg pool + linear projection
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B,128,1,1)
        )
        self.proj = nn.Sequential(
            nn.Flatten(),  # -> (B,128)
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
        self._features_dim = 256

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: th.Tensor) -> th.Tensor:  # type: ignore[override]
        # obs shape (B,C,H,W) expected. If future change -> permute here.
        x = self.cnn(obs)
        x = self.proj(x)
        return x
