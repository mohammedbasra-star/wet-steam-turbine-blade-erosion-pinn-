"""
Multiphase PINN Architecture for Wet Steam Expansion & Blade Erosion.

Approximates:
    (x, y) -> [ Static_Pressure p, Temperature T, Supersaturation S, Droplet_Radius r, Erosion_Rate E_R ]
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class WetSteamPINN(nn.Module):
    """
    PINN approximating wet steam condensation state and erosion fields across LP turbine stage.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 4,
        L_stage: float = 0.45,
        r_hub: float = 0.85,
        r_tip: float = 1.45,
    ) -> None:
        super().__init__()
        self.L_x = L_stage
        self.r_hub = r_hub
        self.r_tip = r_tip

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Output heads:
        # Head 1: Static Pressure p (Pa, range ~ 10,000 to 40,000 Pa)
        # Head 2: Temperature T (K, range ~ 310 to 360 K)
        # Head 3: Supersaturation ratio S (range ~ 1.0 to 4.0)
        # Head 4: Droplet radius r_drop (um)
        # Head 5: Leading-edge erosion rate (mm/year)
        self.head_p = nn.Linear(hidden_dim, 1)
        self.head_T = nn.Linear(hidden_dim, 1)
        self.head_S = nn.Linear(hidden_dim, 1)
        self.head_r = nn.Linear(hidden_dim, 1)
        self.head_er = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = 2.0 * (x / self.L_x) - 1.0
        y_norm = 2.0 * (y - self.r_hub) / (self.r_tip - self.r_hub) - 1.0

        coords = torch.cat([x_norm, y_norm], dim=-1)

        h = torch.tanh(self.in_proj(coords))
        for layer in self.hidden_layers:
            h = h + torch.tanh(layer(h))

        # Pressure from inlet ~35 kPa down to ~12 kPa
        p = 10000.0 + 30000.0 * torch.sigmoid(self.head_p(h))

        # Temperature ~ 315 K to 355 K
        T = 310.0 + 50.0 * torch.sigmoid(self.head_T(h))

        # Supersaturation S >= 1.0
        S = 1.0 + 3.0 * torch.sigmoid(self.head_S(h))

        # Droplet radius (0 to 1.5 um)
        r_drop = 1.5 * torch.sigmoid(self.head_r(h))

        # Erosion rate (0 to 5.0 mm/year)
        er = 5.0 * F.softplus(self.head_er(h))

        return p, T, S, r_drop, er
