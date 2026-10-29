"""Mechanical Specific Energy (Eq. 1)."""

import numpy as np


def compute_mse(WOB, RPM, T, ROP, A_b, eps=1e-6):
    """Compute the Mechanical Specific Energy.

    Parameters
    ----------
    WOB : float or ndarray
        Weight on bit (kN).
    RPM : float or ndarray
        Rotary speed (rpm).
    T : float or ndarray
        Torque (kNm).
    ROP : float or ndarray
        Rate of penetration (m/h).
    A_b : float
        Bit cross-sectional area (m^2).
    eps : float
        Small constant to avoid division by zero in ROP.

    Returns
    -------
    float or ndarray
        MSE in the same units as the input (kN/m^2 equivalent).
    """
    WOB = np.asarray(WOB, dtype=float)
    RPM = np.asarray(RPM, dtype=float)
    T = np.asarray(T, dtype=float)
    ROP = np.asarray(ROP, dtype=float)

    ROP_safe = np.where(np.abs(ROP) < eps, eps, ROP)
    mse = WOB / A_b + 120.0 * np.pi * RPM * T / (ROP_safe * A_b)
    return float(mse) if mse.ndim == 0 else mse
