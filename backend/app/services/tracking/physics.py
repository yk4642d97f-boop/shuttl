"""
Shuttlecock flight dynamics.

A shuttle is dominated by quadratic drag: dv/dt = g - (|v| / (vt^2 / g)) * v,
so it decelerates from 300+ km/h to its terminal velocity (~6.8 m/s) within
metres — a parabola is a poor model. Values from Cohen et al., "The physics of
badminton" (New J. Phys. 2015): drag length L = vt^2 / g ≈ 4.7 m (feather).
"""

from __future__ import annotations

import numpy as np

G = 9.81
TERMINAL_VELOCITY = 6.8                      # m/s, feather shuttle
DRAG_LENGTH = TERMINAL_VELOCITY ** 2 / G     # m
GRAVITY = np.array([0.0, 0.0, -G])


def _accel(v: np.ndarray, drag_length: float) -> np.ndarray:
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    return GRAVITY - speed * v / drag_length


def rk4_step(p: np.ndarray, v: np.ndarray, h: float,
             drag_length: float = DRAG_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """One RK4 step of length h seconds (h may be negative)."""
    k1v = _accel(v, drag_length);                 k1p = v
    k2v = _accel(v + 0.5 * h * k1v, drag_length); k2p = v + 0.5 * h * k1v
    k3v = _accel(v + 0.5 * h * k2v, drag_length); k3p = v + 0.5 * h * k2v
    k4v = _accel(v + h * k3v, drag_length);       k4p = v + h * k3v
    return (p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p),
            v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))


def simulate_state(p0: np.ndarray, v0: np.ndarray, times: np.ndarray,
                   drag_length: float = DRAG_LENGTH,
                   max_dt: float = 1 / 480) -> tuple[np.ndarray, np.ndarray]:
    """Integrate from (p0, v0) at t=0 to each of `times` (seconds, monotonic,
    any sign). Returns (positions (N,3), velocities (N,3))."""
    times = np.asarray(times, dtype=np.float64)
    P, V = np.empty((len(times), 3)), np.empty((len(times), 3))
    p, v, t = np.array(p0, float), np.array(v0, float), 0.0
    for i, target in enumerate(times):
        span = target - t
        n = max(1, int(np.ceil(abs(span) / max_dt)))
        for _ in range(n):
            p, v = rk4_step(p, v, span / n, drag_length)
        t = target
        P[i], V[i] = p, v
    return P, V


def simulate(p0: np.ndarray, v0: np.ndarray, times: np.ndarray,
             drag_length: float = DRAG_LENGTH) -> np.ndarray:
    """Positions only; see simulate_state."""
    return simulate_state(p0, v0, times, drag_length)[0]


def time_to_ground(p0: np.ndarray, v0: np.ndarray, drag_length: float = DRAG_LENGTH,
                   z_ground: float = 0.0, t_max: float = 6.0,
                   dt: float = 1 / 480) -> float | None:
    """Seconds until z crosses z_ground (linear refinement), None if never."""
    p, v, t = np.array(p0, float), np.array(v0, float), 0.0
    while t < t_max:
        p_next, v_next = rk4_step(p, v, dt, drag_length)
        if p_next[2] <= z_ground < p[2] or (p[2] > z_ground >= p_next[2]):
            frac = (p[2] - z_ground) / max(p[2] - p_next[2], 1e-12)
            return t + frac * dt
        p, v, t = p_next, v_next, t + dt
    return None


def kmh(speed_ms: float) -> float:
    return float(speed_ms) * 3.6
