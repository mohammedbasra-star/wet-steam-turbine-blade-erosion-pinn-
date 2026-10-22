"""
Publication-Grade Visualizations for Wet Steam Nucleation & Blade Erosion PINN.

Generates 300 DPI figures:
1. Wilson Line Crossing, Supersaturation Ratio, and Becker-Doring Nucleation Rate.
2. Droplet Radius Growth & Stage Wetness Fraction Trajectories.
3. Spanwise Blade Leading-Edge Water Hammer Shock Pressure & Erosion Rates.
4. Multiphase PINN Composite Loss Optimization Diagnostics.
"""

from typing import Dict, List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
})


def plot_wilson_line_nucleation(
    save_path: str = "figures/wilson_line_nucleation_rate.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from physics.wet_steam_kinetics import WetSteamNucleationPhysics
    physics = WetSteamNucleationPhysics()

    x_axial_cm = np.linspace(0.0, 45.0, 100)  # Stage axial length
    # Pressure expansion from 35 kPa to 12 kPa
    p_pa = (35.0 - 23.0 * (x_axial_cm / 45.0) ** 1.3) * 1e3
    # Supercooled steam temperature
    T_k = 345.0 - 25.0 * (x_axial_cm / 45.0)

    S_vals = []
    J_vals = []

    for p, T in zip(p_pa, T_k):
        s_ratio, _ = physics.supersaturation_and_critical_radius(p, T)
        j_rate = physics.nucleation_rate(p, T)
        S_vals.append(s_ratio)
        J_vals.append(max(j_rate, 1.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Supersaturation S
    ax1.plot(x_axial_cm, S_vals, color="#1f77b4", linewidth=2.5, label="Supersaturation $S = p / p_{\\mathrm{sat}}(T)$")
    ax1.axhline(1.0, color="gray", linestyle=":", label="Saturation Boundary ($S = 1.0$)")
    ax1.axhline(2.8, color="red", linestyle="--", label="Wilson Line Threshold ($S_{\\mathrm{crit}} \\approx 2.8$)")
    ax1.set_xlabel("Stage Axial Coordinate $x$ (cm)")
    ax1.set_ylabel("Supersaturation Ratio $S$")
    ax1.set_title("(a) Thermodynamic Non-Equilibrium Wilson Line Crossing")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Nucleation Rate J
    ax2.semilogy(x_axial_cm, J_vals, color="#d62728", linewidth=2.5, label="Nucleation Rate $J$")
    ax2.set_xlabel("Stage Axial Coordinate $x$ (cm)")
    ax2.set_ylabel(r"Homogeneous Nucleation Rate $J$ ($\mathrm{m^{-3} \cdot s^{-1}}$)")
    ax2.set_title("(b) Becker-Döring Spontaneous Droplet Inception Burst")
    ax2.legend(loc="upper left")
    ax2.grid(True, which="both", alpha=0.3)

    plt.suptitle("Al-Hartha LP Steam Turbine Non-Equilibrium Condensation Inception", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_droplet_growth_wetness(
    save_path: str = "figures/droplet_growth_and_wetness_fraction.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x_axial_cm = np.linspace(15.0, 45.0, 80)

    # Gyarmathy droplet growth profile
    r_drop_um = 0.02 + 0.95 * ((x_axial_cm - 15.0) / 30.0) ** 0.65
    wetness_pct = 0.5 + 9.8 * ((x_axial_cm - 15.0) / 30.0) ** 1.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(x_axial_cm, r_drop_um, color="#2ca02c", linewidth=2.5, label=r"Mean Droplet Radius $r_{\mathrm{drop}}$")
    ax1.set_xlabel("Axial Distance from Wilson Line (cm)")
    ax1.set_ylabel(r"Droplet Radius ($\mu\mathrm{m}$)")
    ax1.set_title(r"(a) Fog Droplet Gyármathy Condensate Growth")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_axial_cm, wetness_pct, color="#17becf", linewidth=2.5, label="Stage Wetness Fraction $Y$ (%)")
    ax2.axhline(10.0, color="darkred", linestyle="--", label="Critical 10% Wetness Limit (Erosion Risk)")
    ax2.set_xlabel("Axial Distance from Wilson Line (cm)")
    ax2.set_ylabel("Wetness Fraction $Y$ (%)")
    ax2.set_title("(b) Liquid Mass Fraction Buildup at L-0 Stage")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Condensate Micro-Droplet Dynamics & Two-Phase Steam Fraction", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_blade_erosion_profile(
    save_path: str = "figures/blade_leading_edge_erosion_profile.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from physics.wet_steam_kinetics import WetSteamNucleationPhysics
    physics = WetSteamNucleationPhysics()

    spans_m = np.linspace(physics.r_hub, physics.r_tip, 60)
    p_impact_gpa = []
    erosion_mm_year = []

    for r in spans_m:
        res = physics.water_hammer_impact_pressure(r)
        p_impact_gpa.append(res["p_impact_gpa"])
        erosion_mm_year.append(res["erosion_rate_mm_year"])

    span_pct = (spans_m - physics.r_hub) / (physics.r_tip - physics.r_hub) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Water Hammer Impact Shock
    ax1.plot(span_pct, p_impact_gpa, color="#d62728", linewidth=2.5, label="Shock Pressure $P_{\\mathrm{impact}}$")
    ax1.axhline(physics.yield_strength_mpa / 1000.0, color="black", linestyle="--", label="Ti-6Al-4V Yield Limit (0.88 GPa)")
    ax1.set_xlabel("Blade Spanwise Height (%) (Hub $\\to$ Tip)")
    ax1.set_ylabel("Water Hammer Shock Pressure (GPa)")
    ax1.set_title("(a) Supersonic Droplet Impact Shockwave")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Heymann Erosion Rate
    ax2.plot(span_pct, erosion_mm_year, color="#7570b3", linewidth=2.5, label="Leading Edge Material Loss (mm/year)")
    ax2.axvline(80.0, color="red", linestyle=":", label="Critical 80% Tip Erosion Region")
    ax2.set_xlabel("Blade Spanwise Height (%) (Hub $\\to$ Tip)")
    ax2.set_ylabel("Erosion Depth Rate (mm/year)")
    ax2.set_title("(b) Annual Water Droplet Erosion (WDE) Depth")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Ti-6Al-4V Low-Pressure Steam Turbine Blade Erosion Hazard Map", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pinn_convergence(
    epochs: List[int],
    total_loss: List[float],
    flow_loss: List[float],
    erosion_loss: List[float],
    save_path: str = "figures/pinn_wet_steam_loss_convergence.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(epochs, total_loss, color="#1f77b4", linewidth=2.2, label="Total Multiphase Loss")
    ax.semilogy(epochs, flow_loss, color="#d62728", linestyle="--", linewidth=2.0, label="Steam Expansion Residual Loss")
    ax.semilogy(epochs, erosion_loss, color="#2ca02c", linestyle="-.", linewidth=2.0, label="Droplet Erosion Law Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Residual")
    ax.set_title("Wet Steam Multiphase PINN Optimization Diagnostics", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
