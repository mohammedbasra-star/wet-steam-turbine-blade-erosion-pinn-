"""
Training Pipeline for Al-Hartha LP Steam Turbine Blade Erosion PINN.
"""

from typing import Dict, List, Tuple
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from physics.wet_steam_kinetics import WetSteamNucleationPhysics
from data.dataset import WetSteamCollocationDataset
from models.wet_steam_pinn import WetSteamPINN
from utils.visualization import (
    plot_wilson_line_nucleation,
    plot_droplet_growth_wetness,
    plot_blade_erosion_profile,
    plot_pinn_convergence,
)


def run_training(
    epochs: int = 35,
    lr: float = 3e-3,
    n_interior: int = 2500,
    n_boundary: int = 400,
    device_name: str = "cpu",
    output_dir: str = "checkpoints",
    figures_dir: str = "figures",
) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, output_dir)
    figures_dir = os.path.join(base_dir, figures_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device(device_name if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    print(f"=== Al-Hartha LP Steam Turbine Blade Erosion PINN Training ===")
    print(f"Device: {device} | Epochs: {epochs} | Interior Points: {n_interior}")

    physics = WetSteamNucleationPhysics()
    dataset = WetSteamCollocationDataset(
        n_interior=n_interior,
        n_boundary=n_boundary,
        L_stage_m=0.45,
        r_hub_m=physics.r_hub,
        r_tip_m=physics.r_tip,
    )

    model = WetSteamPINN(
        in_dim=2,
        hidden_dim=64,
        num_layers=4,
        L_stage=0.45,
        r_hub=physics.r_hub,
        r_tip=physics.r_tip,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    total_losses = []
    flow_losses = []
    erosion_losses = []

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        batches = dataset.get_batches()
        x_int, y_int = batches["interior"][0].to(device), batches["interior"][1].to(device)

        p, T, S, r_drop, er = model(x_int, y_int)

        # 1. Non-equilibrium expansion: dp/dx < 0 (favorable pressure gradient)
        dp_dx = torch.autograd.grad(p, x_int, torch.ones_like(p), create_graph=True)[0]
        # Soft penalty on positive pressure gradient (expansion requires dp/dx < 0)
        res_dp = F.relu(dp_dx + 1000.0)

        # 2. Water hammer erosion scaling with spanwise radius y
        # Velocity v_rel increases linearly with radius y -> erosion scales as y^5
        v_rel_span = torch.sqrt((physics.omega * y_int) ** 2 + (220.0 ** 2))
        er_target = 0.45 * ((v_rel_span / 180.0) ** 4.5)
        res_erosion = er - er_target

        loss_pde = 1e-4 * torch.mean(res_dp ** 2) + torch.mean(res_erosion ** 2)

        # 3. Boundaries:
        # Inlet (x = 0): p = 35 kPa, T = 345 K
        x_in, y_in = batches["inlet"][0].to(device), batches["inlet"][1].to(device)
        p_in, T_in, _, r_in, er_in = model(x_in, y_in)
        loss_inlet = 1e-6 * torch.mean((p_in - 35000.0) ** 2) + torch.mean((T_in - 345.0) ** 2) + torch.mean(r_in ** 2) + torch.mean(er_in ** 2)

        # Outlet (x = L_x): p = 12 kPa (Al-Hartha summer condenser)
        x_out, y_out = batches["outlet"][0].to(device), batches["outlet"][1].to(device)
        p_out, _, _, _, _ = model(x_out, y_out)
        loss_outlet = 1e-6 * torch.mean((p_out - 12000.0) ** 2)

        loss_bc = 10.0 * loss_inlet + 10.0 * loss_outlet
        loss_total = loss_pde + loss_bc

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        total_losses.append(loss_total.item())
        flow_losses.append(torch.mean(res_dp ** 2).item())
        erosion_losses.append(torch.mean(res_erosion ** 2).item())

        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_loss": best_loss,
                },
                os.path.join(output_dir, "best_wet_steam_pinn_model.pt"),
            )

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Loss: {loss_total.item():.4f} (PDE: {loss_pde.item():.4f}, BC: {loss_bc.item():.4f})"
            )

    elapsed = time.time() - start_time
    print(f"PINN Training completed in {elapsed:.2f} s. Best Loss: {best_loss:.4f}")

    # Generate Figures
    print("\nGenerating publication figures...")
    plot_wilson_line_nucleation(
        save_path=os.path.join(figures_dir, "wilson_line_nucleation_rate.png")
    )
    plot_droplet_growth_wetness(
        save_path=os.path.join(figures_dir, "droplet_growth_and_wetness_fraction.png")
    )
    plot_blade_erosion_profile(
        save_path=os.path.join(figures_dir, "blade_leading_edge_erosion_profile.png")
    )
    plot_pinn_convergence(
        epochs=list(range(1, epochs + 1)),
        total_loss=total_losses,
        flow_loss=flow_losses,
        erosion_loss=erosion_losses,
        save_path=os.path.join(figures_dir, "pinn_wet_steam_loss_convergence.png"),
    )
    print("All 4 publication figures generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_training(epochs=args.epochs, lr=args.lr, device_name=args.device)
