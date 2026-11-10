"""Risk-weighted benefit (roadmap 6): expected collisions with active satellites.

A debris object inside shell s collides with the shell's active satellites at
rate k_s = A_s * v_rel / V_s (kinetic-gas estimate), with A_s the summed
collision cross-section of the actives in the shell, v_rel a mean relative
speed and V_s = 4 pi r^2 dh the shell volume. Expected collisions of an engaged
object = sum_s k_s * (time it spends in s), on its natural path and on its
lasered path (shell residence from the lifetime integrator, to re-entry or the
200-yr horizon). Collisions avoided = natural - lasered, summed over every
engaged object: a removal counts by the risk it carries, and lowering an object
through a busy shell counts against the laser.

Active populations and cross-sections are PICKS ([risk] in the scenario).
"""

from __future__ import annotations

import numpy as np

from .constants import R_EARTH, YEAR


def shell_rates_per_yr(risk_cfg) -> dict[str, float]:
    """Collision rate [1/yr] of one debris object inside each shell."""
    out = {}
    for name, lo, hi, area in zip(risk_cfg.shell_names, risk_cfg.shell_lo_m, risk_cfg.shell_hi_m,
                                  risk_cfg.shell_active_area_m2):
        r = R_EARTH + 0.5 * (lo + hi)
        vol = 4.0 * np.pi * r**2 * (hi - lo)
        out[name] = area * risk_cfg.v_rel_m_s / vol * YEAR
    return out


def evaluate(sc, summary: dict) -> dict:
    """Expected collisions with active satellites, natural vs lasered, for one run's summary."""
    if not sc.risk.enabled:
        return {}
    rates = shell_rates_per_yr(sc.risk)
    nat = aft = 0.0
    out = {}
    for name, k in rates.items():
        n = summary.get(f"risk_{name}_obj_yr_natural")
        a = summary.get(f"risk_{name}_obj_yr_after")
        if n is None or a is None:
            continue
        out[f"risk_{name}_collisions_natural"] = k * n
        out[f"risk_{name}_collisions_after"] = k * a
        nat += k * n
        aft += k * a
    yr = summary.get("sim_hours", 0.0) * 3600.0 / YEAR
    out["active_collisions_natural"] = nat
    out["active_collisions_avoided"] = nat - aft
    out["active_collisions_avoided_per_yr"] = (nat - aft) / yr if yr > 0 else np.nan
    return out
