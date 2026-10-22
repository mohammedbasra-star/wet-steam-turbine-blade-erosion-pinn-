"""
Non-Equilibrium Wet Steam Nucleation & Blade Erosion Physics.

Models spontaneous condensation and water droplet impact erosion (WDE) in the
last stages (L-0 / L-1) of low-pressure steam turbines (Al-Hartha Power Station):
- Wilson line crossing & supersaturation ratio S = p / p_sat(T)
- Becker-Doring nucleation rate J(T, S) and critical droplet radius r*
- Gyarmathy droplet growth kinetics dr/dt
- Supersonic water hammer impact shock pressure P_impact on Ti-6Al-4V blades
- Heymann erosion rate and leading-edge mass removal
"""

from typing import Dict, Tuple
import numpy as np


class WetSteamNucleationPhysics:
    """
    Thermodynamics of non-equilibrium wet steam condensation and droplet erosion.
    """

    def __init__(
        self,
        blade_rpm: float = 3000.0,    # Rotor speed (RPM)
        tip_radius_m: float = 1.45,   # L-0 blade tip radius (1.45 m)
        hub_radius_m: float = 0.85,   # L-0 blade hub radius (0.85 m)
        inlet_pressure_bar: float = 0.35, # LP turbine stage inlet pressure (bar)
        condenser_p_bar: float = 0.12,# Al-Hartha summer condenser backpressure (0.12 bar)
    ) -> None:
        self.rpm = blade_rpm
        self.omega = 2.0 * np.pi * blade_rpm / 60.0 # 314.16 rad/s
        self.r_tip = tip_radius_m
        self.r_hub = hub_radius_m
        self.u_tip = self.omega * tip_radius_m # ~455 m/s tip peripheral speed
        self.p_in = inlet_pressure_bar * 1e5
        self.p_cond = condenser_p_bar * 1e5

        # Steam & water properties
        self.R_v = 461.5       # Gas constant for water vapor (J/kg*K)
        self.gamma_v = 1.30    # Isentropic expansion coefficient for superheated steam
        self.rho_L = 980.0     # Liquid water density at low pressure (kg/m^3)
        self.c_L = 1480.0      # Speed of sound in liquid water (m/s)
        self.sigma = 0.0589    # Surface tension of water (N/m)
        self.h_fg = 2.40e6     # Enthalpy of vaporization (J/kg)
        self.k_B = 1.380649e-23# Boltzmann constant (J/K)
        self.m_molecule = 2.99e-26 # Mass of single H2O molecule (kg)

        # Ti-6Al-4V Titanium blade properties
        self.yield_strength_mpa = 880.0
        self.hardness_hv = 350.0

    def saturation_pressure(self, T_k: float) -> float:
        """Magnus-Tetens formula for low pressure steam saturation (Pa)."""
        T_c = T_k - 273.15
        return 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))

    def saturation_temperature(self, p_pa: float) -> float:
        """Inverse saturation temperature T_sat(p) in Kelvin."""
        p_clamped = np.clip(p_pa, 1000.0, 500000.0)
        # Antoine formulation approximation
        ln_p = np.log(p_clamped / 100.0)
        T_c = 237.3 * ln_p / (17.27 - ln_p)
        return float(T_c + 273.15)

    def supersaturation_and_critical_radius(
        self, p_pa: float, T_k: float
    ) -> Tuple[float, float]:
        """
        Calculates:
            Supersaturation ratio: S = p / p_sat(T)
            Critical Kelvin radius: r* = (2 * sigma) / (rho_L * R_v * T * ln(S))
        """
        p_sat = self.saturation_pressure(T_k)
        S = max(p_pa / max(p_sat, 1e-3), 1.001)

        ln_S = np.log(S)
        r_star = (2.0 * self.sigma) / (self.rho_L * self.R_v * T_k * ln_S)
        return float(S), float(r_star)

    def nucleation_rate(self, p_pa: float, T_k: float) -> float:
        """
        Classical Becker-Doring homogeneous nucleation rate J (nuclei / m^3 * s).
        """
        S, r_star = self.supersaturation_and_critical_radius(p_pa, T_k)
        if S < 1.15:
            return 0.0  # Above Wilson line, zero homogeneous nucleation

        rho_v = p_pa / (self.R_v * T_k)
        prefactor = np.sqrt(2.0 * self.sigma / (np.pi * (self.m_molecule ** 3))) * ((rho_v ** 2) / self.rho_L)

        # Free energy barrier Delta G* = (4/3) * pi * r*^2 * sigma
        delta_G = (4.0 / 3.0) * np.pi * (r_star ** 2) * self.sigma
        exponent = -delta_G / (self.k_B * T_k)

        # Wilson line condensation threshold (S ~ 2.5 to 3.5)
        J = prefactor * np.exp(np.clip(exponent, -60.0, 60.0))
        return float(J)

    def droplet_growth_rate(self, r: float, p_pa: float, T_k: float) -> float:
        """
        Gyarmathy droplet growth rate dr/dt (m/s).
        """
        T_sat = self.saturation_temperature(p_pa)
        delta_T = max(T_sat - T_k, 0.0)  # Supercooling degree
        if delta_T <= 0.0:
            return 0.0

        # Vapor thermal conductivity ~ 0.025 W/m*K
        k_v = 0.025
        # Knudsen number Kn = mean_free_path / (2 * r)
        kn = 0.15e-6 / (2.0 * max(r, 1e-9))

        dr_dt = (k_v * delta_T) / (self.rho_L * self.h_fg * max(r, 1e-8) * (1.0 + 3.18 * kn))
        return float(min(dr_dt, 0.02))

    def water_hammer_impact_pressure(self, r_span_m: float) -> Dict[str, float]:
        """
        Calculates supersonic liquid droplet impact shock pressure on the moving blade:
            P_impact = rho_L * C_L * v_rel * (1 + 2 * v_rel / C_L)
        """
        r_span = np.clip(r_span_m, self.r_hub, self.r_tip)
        u_blade = self.omega * r_span # Blade speed (m/s)

        # Steam axial velocity ~ 220 m/s
        v_axial = 220.0
        # Droplet relative impact velocity on leading edge
        v_rel = np.sqrt(u_blade ** 2 + v_axial ** 2)

        # Classical Water Hammer shock equation (with acoustic impedance correction)
        p_impact_pa = self.rho_L * self.c_L * v_rel * (1.0 + 2.0 * (v_rel / self.c_L))
        p_impact_gpa = p_impact_pa / 1e9

        # Heymann erosion rate (mm of blade thickness eroded per 10,000 hours)
        # E_R proportional to (v_rel / v_threshold)^5.2
        v_threshold = 180.0  # Safe impact velocity threshold for Titanium
        if v_rel > v_threshold:
            erosion_rate_mm_year = 0.45 * ((v_rel / v_threshold) ** 5.0)
        else:
            erosion_rate_mm_year = 0.01

        return {
            "r_span_m": float(r_span),
            "blade_speed_m_s": float(u_blade),
            "v_relative_m_s": float(v_rel),
            "p_impact_gpa": float(p_impact_gpa),
            "erosion_rate_mm_year": float(erosion_rate_mm_year),
            "is_above_yield": bool(p_impact_gpa * 1000.0 >= self.yield_strength_mpa),
        }
