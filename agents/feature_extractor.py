#!/usr/bin/env python3
"""
SLAMFeatureExtractor: Custom CNN+Scalar feature extractor for Dict observation space.

Architecture (matches revised 3-layer design):
  Input: Dict with "map_tensor" (3, 64, 64) and "scalars" (8,)

  CNN Branch:
    - CoordConv: prepend 2 normalized coordinate channels → (5, 64, 64)
    - Conv2d(5→32, k=5, s=2, p=2) + BatchNorm + ReLU → (32, 32, 32)
    - Conv2d(32→64, k=3, s=2, p=1) + BatchNorm + ReLU → (64, 16, 16)
    - Conv2d(64→64, k=3, s=2, p=1) + BatchNorm + ReLU → (64, 8, 8)
    - Flatten → 4096

  Scalar Branch:
    - LayerNorm(8) → Linear(8, 64) + ReLU → Linear(64, 32) + ReLU → 32

  Combine:
    - Concat(4096, 32) = 4128
    - Linear(4128, 512) + ReLU
    - Linear(512, features_dim)    ← NO final ReLU (SB3 standard practice)

Fixes from audit:
  - features_dim=512 (was 128 — 16x compression bottleneck)
  - No final ReLU on combine output (was clipping negative features)
  - 2-layer scalar net with LayerNorm (was 1-layer, no normalization)
  - BatchNorm2d after each conv layer (was missing, caused unstable training)
  - CoordConv: 2 coordinate channels prepended (spatial info was destroyed by Flatten)
  - Orthogonal weight initialization (better for RL than default Kaiming)
  - .float() dtype guard on both inputs (prevents float64/float32 mismatch crash)
  - Conv1 channels increased to 32 (was 16 — limited feature vocabulary)
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class SLAMFeatureExtractor(BaseFeaturesExtractor):
    """CNN for map tensor + MLP for scalars, concatenated.

    Designed for the Active SLAM Dict observation space:
      - "map_tensor": (3, 64, 64) — occupancy, visited, trajectory channels
      - "scalars": (8,) — covariance, frontier count/dist/dir, coverage,
                          altitude, position, step fraction
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        # Must call super with the final output dimension
        super().__init__(observation_space, features_dim)

        map_shape = observation_space["map_tensor"].shape  # (3, 64, 64)
        n_channels = map_shape[0]                          # 3
        n_scalars = observation_space["scalars"].shape[0]   # 8

        # CoordConv adds 2 channels (x_coords, y_coords) → total input channels = 5
        cnn_input_channels = n_channels + 2

        # ============================================
        # CNN Branch (with BatchNorm)
        # ============================================
        self.cnn = nn.Sequential(
            # Layer 1: (5, 64, 64) → (32, 32, 32)
            nn.Conv2d(cnn_input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 2: (32, 32, 32) → (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 3: (64, 16, 16) → (64, 8, 8)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Flatten: (64, 8, 8) → 4096
            nn.Flatten(),
        )

        # Calculate CNN output size dynamically (accounts for any map_shape)
        with torch.no_grad():
            dummy = torch.zeros(1, cnn_input_channels, map_shape[1], map_shape[2])
            cnn_out_size = self.cnn(dummy).shape[1]  # should be 4096

        # ============================================
        # Scalar Branch (with LayerNorm)
        # ============================================
        # LayerNorm normalizes the heterogeneous scalar inputs
        # (covariance ~0-1, coverage ~0-1, altitude ~0-1, etc.)
        # Two layers so the network can learn conjunctive conditions like
        # "high covariance AND late in episode → be conservative"
        self.scalar_net = nn.Sequential(
            nn.LayerNorm(n_scalars),
            nn.Linear(n_scalars, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # ============================================
        # Combine Branch (NO final ReLU)
        # ============================================
        # Intermediate layer reduces 4128 → 512 with ReLU,
        # then 512 → features_dim WITHOUT activation.
        # SB3's Actor and Critic heads expect unrestricted feature values
        # (the Critic must estimate negative values for bad states).
        self.combine = nn.Sequential(
            nn.Linear(cnn_out_size + 32, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            # NO ReLU here — SB3 standard practice
        )

        # ============================================
        # Orthogonal Weight Initialization
        # ============================================
        # Better than default Kaiming for RL (PPO paper recommendation)
        self._init_weights()

        # Pre-compute coordinate grids (will be expanded to batch size in forward)
        # These are registered as buffers so they move to GPU with the model
        h, w = map_shape[1], map_shape[2]
        x_coords = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(1, 1, h, w)
        y_coords = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(1, 1, h, w)
        self.register_buffer("x_coords", x_coords)
        self.register_buffer("y_coords", y_coords)

    def _init_weights(self):
        """Apply orthogonal initialization to all linear and conv layers."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observations):
        """Forward pass through both branches + combine.

        Args:
            observations: dict with
                "map_tensor": (B, 3, 64, 64) float32
                "scalars": (B, 8) float32

        Returns:
            features: (B, features_dim) float32
        """
        map_tensor = observations["map_tensor"].float()
        scalars = observations["scalars"].float()

        # ---- CoordConv: append x, y coordinate channels ----
        B = map_tensor.shape[0]
        # Expand pre-computed coords to match batch size
        x_exp = self.x_coords.expand(B, -1, -1, -1)
        y_exp = self.y_coords.expand(B, -1, -1, -1)
        # Concatenate: (B, 3, 64, 64) + (B, 1, 64, 64) + (B, 1, 64, 64) → (B, 5, 64, 64)
        map_with_coords = torch.cat([map_tensor, x_exp, y_exp], dim=1)

        # ---- CNN branch ----
        cnn_features = self.cnn(map_with_coords)

        # ---- Scalar branch ----
        scalar_features = self.scalar_net(scalars)

        # ---- Combine ----
        combined = torch.cat([cnn_features, scalar_features], dim=1)
        return self.combine(combined)