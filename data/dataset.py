"""
Collocation Point Dataset for Low-Pressure Turbine Stage Cascade.
"""

from typing import Dict, Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class WetSteamCollocationDataset(Dataset):
    """
    Collocation points across LP steam turbine stage:
    Axial coordinate x in [0.0, 0.45] m (nozzle stator to blade trailing edge)
    Spanwise coordinate y in [r_hub, r_tip] = [0.85, 1.45] m
    """

    def __init__(
        self,
        n_interior: int = 2500,
        n_boundary: int = 400,
        L_stage_m: float = 0.45,
        r_hub_m: float = 0.85,
        r_tip_m: float = 1.45,
    ) -> None:
        self.L_x = L_stage_m
        self.r_hub = r_hub_m
        self.r_tip = r_tip_m
        self.n_int = n_interior
        self.n_bnd = n_boundary

        self.resample()

    def resample(self) -> None:
        # Interior points
        x_int = np.random.uniform(0.0, self.L_x, self.n_int).astype(np.float32)
        y_int = np.random.uniform(self.r_hub, self.r_tip, self.n_int).astype(np.float32)
        self.x_int = torch.tensor(x_int, requires_grad=True).unsqueeze(1)
        self.y_int = torch.tensor(y_int, requires_grad=True).unsqueeze(1)

        # Stage inlet (x = 0): Superheated steam
        y_inlet = np.random.uniform(self.r_hub, self.r_tip, self.n_bnd).astype(np.float32)
        x_inlet = np.zeros_like(y_inlet)
        self.x_in = torch.tensor(x_inlet, requires_grad=True).unsqueeze(1)
        self.y_in = torch.tensor(y_inlet, requires_grad=True).unsqueeze(1)

        # Stage outlet / Condenser entrance (x = L_x): Fully expanded wet steam
        y_outlet = np.random.uniform(self.r_hub, self.r_tip, self.n_bnd).astype(np.float32)
        x_outlet = np.full_like(y_outlet, self.L_x)
        self.x_out = torch.tensor(x_outlet, requires_grad=True).unsqueeze(1)
        self.y_out = torch.tensor(y_outlet, requires_grad=True).unsqueeze(1)

        # Hub (y = r_hub) and Tip (y = r_tip) boundaries
        x_wall = np.random.uniform(0.0, self.L_x, self.n_bnd).astype(np.float32)
        self.x_wall = torch.tensor(x_wall, requires_grad=True).unsqueeze(1)
        self.y_hub = torch.full_like(self.x_wall, self.r_hub, requires_grad=True)
        self.y_tip = torch.full_like(self.x_wall, self.r_tip, requires_grad=True)

    def get_batches(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return {
            "interior": (self.x_int, self.y_int),
            "inlet": (self.x_in, self.y_in),
            "outlet": (self.x_out, self.y_out),
            "hub": (self.x_wall, self.y_hub),
            "tip": (self.x_wall, self.y_tip),
        }
