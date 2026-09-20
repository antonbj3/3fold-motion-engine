#!/usr/bin/env python3
"""Geotechnical bearing capacity formulas and Ny table.

Implements:
- Terzaghi (1943)
- Meyerhof (1963)
- Vesic (1973)
- Hansen (1970)
Both for general shear failure (phi) and local/punching shear failure (reduced phi* = atan(2/3 tan phi)).
"""
import math
import numpy as np

def calc_bearing_factors(phi_deg: float, local_shear: bool = False):
    """Calculate Nq, Nc, Ny according to Terzaghi, Meyerhof, Vesic, and Hansen."""
    phi_eff = phi_deg
    if local_shear:
        # Vesic / Terzaghi reduced friction angle for loose / compressible sand
        phi_eff = math.degrees(math.atan(2.0 / 3.0 * math.tan(math.radians(phi_deg))))
        
    p = math.radians(phi_eff)
    tan_p = math.tan(p)
    
    # Nq (Prandtl 1920 / Reissner 1924)
    nq = math.exp(math.pi * tan_p) * (math.tan(math.radians(45.0 + phi_eff / 2.0))**2)
    
    # Nc (Prandtl)
    nc = (nq - 1.0) / tan_p if abs(tan_p) > 1e-9 else 0.0
    
    # Ny variations:
    # 1. Meyerhof (1963) / Terzaghi approximation used in U24/U86:
    ny_meyerhof = (nq - 1.0) * math.tan(1.4 * p)
    
    # 2. Vesic (1973):
    ny_vesic = 2.0 * (nq + 1.0) * tan_p
    
    # 3. Hansen (1970):
    ny_hansen = 1.5 * (nq - 1.0) * tan_p
    
    # 4. Kumbhojkar (1993) rigorous numerical integration of Terzaghi's rough footing:
    # At 30 deg: Kumbhojkar Ny = 19.13; at 35 deg: 41.1; at 40 deg: 93.7.
    # Terzaghi (1943) original table:
    # phi = 20: Ny ~ 5.4, Nq ~ 7.4
    # phi = 25: Ny ~ 9.7, Nq ~ 12.7
    # phi = 30: Ny ~ 19.7, Nq ~ 22.5
    # phi = 35: Ny ~ 42.4, Nq ~ 41.4
    # phi = 40: Ny ~ 100.4, Nq ~ 81.3
    # A standard fit for Terzaghi original table:
    ny_terzaghi = (nq - 1.0) * math.tan(1.4 * p) # standard code implementation
    
    return {
        "phi_input_deg": phi_deg,
        "local_shear": local_shear,
        "phi_eff_deg": phi_eff,
        "Nq": nq,
        "Nc": nc,
        "Ny_Meyerhof": ny_meyerhof,
        "Ny_Vesic": ny_vesic,
        "Ny_Hansen": ny_hansen,
        "Ny_Terzaghi": ny_terzaghi
    }

def bearing_capacity_qu(phi_deg: float, gamma: float = 9810.0, B: float = 0.06, depth: float = 0.025,
                        formula: str = "Vesic", local_shear: bool = False, is_square: bool = False):
    """Compute ultimate bearing capacity qu [Pa] and total load [N]."""
    fac = calc_bearing_factors(phi_deg, local_shear=local_shear)
    nq = fac["Nq"]
    ny = fac[f"Ny_{formula}"]
    
    if is_square:
        # Terzaghi shape factor: 0.4 for gamma term, 1.0 for q term
        qu = 0.4 * gamma * B * ny + gamma * depth * nq
        F_total = qu * (B * B)
    else:
        # Continuous strip footing: 0.5 for gamma term, 1.0 for q term
        qu = 0.5 * gamma * B * ny + gamma * depth * nq
        F_total = qu * (B * B) # normalized for length L = B
    return qu, F_total, fac

def print_ny_comparison():
    print("=" * 90)
    print("TABLE: BEARING CAPACITY FACTOR Ny AND CAPACITY AT phi = 20, 25, 30, 35, 40 deg")
    print("=" * 90)
    print(f"{'phi [deg]':>10s} | {'Nq':>8s} | {'Ny (Meyerhof)':>14s} | {'Ny (Vesic)':>12s} | {'Ny (Hansen)':>12s} | {'Ny_Vesic/Ny_Hansen':>18s}")
    print("-" * 90)
    for phi in [20.0, 23.0, 25.0, 30.0, 35.0, 40.0]:
        fac = calc_bearing_factors(phi, local_shear=False)
        ratio = fac['Ny_Vesic'] / fac['Ny_Hansen'] if fac['Ny_Hansen'] > 0 else 0
        print(f"{phi:10.1f} | {fac['Nq']:8.2f} | {fac['Ny_Meyerhof']:14.2f} | {fac['Ny_Vesic']:12.2f} | {fac['Ny_Hansen']:12.2f} | {ratio:18.2f}x")
        
    print("\n" + "=" * 90)
    print("TABLE: LOCAL SHEAR FAILURE (VESIC REDUCED phi* = atan(2/3 tan phi)) FOR LOOSE SAND")
    print("=" * 90)
    print(f"{'phi [deg]':>10s} | {'phi* [deg]':>10s} | {'Nq*':>8s} | {'Ny* (Meyerhof)':>14s} | {'Ny* (Vesic)':>12s} | {'Ny* (Hansen)':>12s}")
    print("-" * 90)
    for phi in [20.0, 23.0, 25.0, 30.0, 35.0, 40.0]:
        fac = calc_bearing_factors(phi, local_shear=True)
        print(f"{phi:10.1f} | {fac['phi_eff_deg']:10.2f} | {fac['Nq']:8.2f} | {fac['Ny_Meyerhof']:14.2f} | {fac['Ny_Vesic']:12.2f} | {fac['Ny_Hansen']:12.2f}")

if __name__ == "__main__":
    print_ny_comparison()
