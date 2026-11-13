#!/usr/bin/env python
"""Exact prefilter for rally._incoming_side_reading. Under the drag ODE
(dv/dt = -g e_z - |v| v / L): vx never changes sign, d|vx|/dt = -|v||vx|/L;
|v| <= u(t), the 1-D Riccati solution u' = g - u^2/L from u(0) = |v0| (terminal
vt = sqrt(g L)), and |v| >= |vx|. Hence the x-travel D satisfies
  L ln(1 + |vx0| t / L) >= D >= |vx0| * int_0^t exp(-int u / L).
If the whole bracket (+- margin for the RK4 error) is on one side of the dead
zone the reading is decided without integrating. Checked here against RK4.

  .venv/bin/python -m scripts.accuracy.xbracket
"""
from __future__ import annotations

import math

from app.services.tracking import rally as R
from app.services.tracking.physics import G
from scripts.accuracy.common import load_samples, sim_args
from scripts.accuracy.side_trans import prepared


def x_bracket(seg, t: float) -> tuple[float, float]:
    x0, vx0 = float(seg.p0[0]), float(seg.v0[0])
    tau = t - float(seg.t0)
    L = float(seg.drag_length or R.DRAG_LENGTH)
    ax = abs(vx0)
    if ax == 0.0 or tau <= 0.0:
        return x0, x0
    d_hi = L * math.log1p(ax * tau / L)
    sp = math.sqrt(float(seg.v0[0]) ** 2 + float(seg.v0[1]) ** 2 + float(seg.v0[2]) ** 2)
    vt = math.sqrt(G * L)
    k = vt / L
    # exp(-int_0^s u/L) for the Riccati u; D_lo = ax * int_0^tau of it (closed forms)
    if sp > vt:
        c = math.atanh(vt / sp)                    # u = vt coth(k s + c)
        # exp(-int u/L) = sinh(c) / sinh(k s + c); int_0^tau = sinh(c)/k * ln(tanh((k tau + c)/2) / tanh(c/2))
        d_lo = ax * math.sinh(c) / k * math.log(math.tanh((k * tau + c) / 2) / math.tanh(c / 2))
    else:
        d_lo = ax * (1.0 - math.exp(-k * tau)) / k  # u <= vt: exp(-vt s / L) is a lower bound
    sgn = 1.0 if vx0 > 0 else -1.0
    a, b = x0 + sgn * d_lo, x0 + sgn * d_hi
    return (a, b) if a <= b else (b, a)


def main():
    cfg = R.RallyConfig()
    n = dec = bad = 0
    worst = 0.0
    for style in ("behind", "random"):
        for seed in (0, 1000):
            for d in load_samples(sim_args(style, "1920x1080", 60.0, 5), list(range(seed, seed + 40)), 4):
                mo, hs, segs, conts, groups, truth = prepared(d, cfg)
                for i in range(1, len(hs)):
                    prev = conts[i - 1] if conts[i - 1] is not None else segs[i - 1]
                    h = hs[i]
                    if prev is None or not float(prev.t0) < h.t:
                        continue
                    t_cov = float(mo.t[min(max(int(prev.end_frame), 0), mo.n - 1)]) + 0.25
                    if h.t > t_cov:
                        continue
                    n += 1
                    x = R._state_at(prev, h.t)[0]
                    lo, hi = x_bracket(prev, h.t)
                    worst = max(worst, lo - x, x - hi)
                    m = 0.05
                    fast = None
                    if lo - m > cfg.side_x_dead:
                        fast = "far"
                    elif hi + m < -cfg.side_x_dead:
                        fast = "near"
                    elif lo - m >= -cfg.side_x_dead and hi + m <= cfg.side_x_dead:
                        fast = "none"
                    if fast is not None:
                        dec += 1
                        ref = "near" if x < -cfg.side_x_dead else ("far" if x > cfg.side_x_dead else "none")
                        bad += fast != ref
    print(f"n={n} decided by bracket {dec} ({dec / n:.1%}) disagreements {bad}; max violation of the bracket by RK4 {worst:.2e} m")


if __name__ == "__main__":
    main()
