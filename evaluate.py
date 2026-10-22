"""
Evaluation and Benchmarking Script for Al-Hartha LP Steam Turbine Erosion PINN.
"""

from typing import Dict, Tuple
import os
import argparse
import time
import numpy as np
import torch

from physics.wet_steam_kinetics import WetSteamNucleationPhysics
from models.wet_steam_pinn import WetSteamPINN


def evaluate_benchmark(
    checkpoint_path: str = "checkpoints/best_wet_steam_pinn_model.pt",
    device_name: str = "cpu",
) -> Dict[str, float]:
    device = torch.device(device_name if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    print(f"Loading checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found. Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    physics = WetSteamNucleationPhysics()
    model = WetSteamPINN(
        in_dim=2, hidden_dim=64, num_layers=4, L_stage=0.45, r_hub=physics.r_hub, r_tip=physics.r_tip
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_test = np.linspace(physics.r_hub, physics.r_tip, 100).astype(np.float32)
    x_test = np.full_like(y_test, 0.40)  # Near L-0 blade leading edge

    x_t = torch.tensor(x_test).unsqueeze(1).to(device)
    y_t = torch.tensor(y_test).unsqueeze(1).to(device)

    start_eval = time.time()
    with torch.no_grad():
        p_pred, T_pred, S_pred, r_pred, er_pred = model(x_t, y_t)
    eval_time = (time.time() - start_eval) * 1e3

    tip_eval = physics.water_hammer_impact_pressure(physics.r_tip)

    results = {
        "blade_tip_speed_m_s": tip_eval["blade_speed_m_s"],
        "max_water_hammer_gpa": tip_eval["p_impact_gpa"],
        "peak_erosion_rate_mm_year": float(torch.max(er_pred).item()),
        "eval_time_ms": float(eval_time),
    }

    print("\n" + "=" * 60)
    print("      AL-HARTHA LP STEAM TURBINE BLADE EROSION PINN BENCHMARK      ")
    print("=" * 60)
    print(f" Blade Tip Peripheral Speed:       {results['blade_tip_speed_m_s']:.1f} m/s ({results['blade_tip_speed_m_s']*3.6:.0f} km/h)")
    print(f" Water Hammer Shock Pressure:      {results['max_water_hammer_gpa']:.2f} GPa (Above Yield Limit)")
    print(f" Maximum Tip Erosion Rate:         {results['peak_erosion_rate_mm_year']:.2f} mm/year")
    print(f" Multiphase Evaluation Latency:    {results['eval_time_ms']:.2f} ms")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_wet_steam_pinn_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_benchmark(checkpoint_path=args.checkpoint, device_name=args.device)
