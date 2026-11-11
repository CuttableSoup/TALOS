"""Focus campaign: concentrate bursts on a fixed set of large objects.

Every mission eventually faces the large, heavy debris: once the light tail is gone
(README "2-year design run"), what is left is mu >= 2 kg/m^2, and greedy scheduling spreads
its bursts over hundreds of such objects (~5 m/s each per month against ~150 m/s needed),
so none are removed. A campaign picks a focus set at the start of the run (for a
fresh-platform restart: at the restart) and gives it priority:

* ``exclusive = true``: platforms fire only at focus objects (the large-debris test:
  what the design can do when it devotes itself to them);
* every focus burst must predict >= ``min_impulse_frac`` x the impulse of a full burst at the
  Cm-peak fluence, or the platform waits for the range to close. With a small set the
  platforms are opportunity-limited (an object is engaged at most once per chunk), so a strict
  gate trades away more passes than it gains per burst: 0.15 (README "Large-debris test").
* ``exclusive = false``: a focus candidate is taken whenever its predicted impulse per area
  (dv x mu, i.e. burst geometry, independent of the object's mass) is at least
  ``quality`` x the best candidate's; otherwise the run's normal score applies.

Among focus candidates the run score (default removal_progress) picks, so bursts
concentrate on the focus objects already closest to removal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scheduler import dv_to_remove


@dataclass(frozen=True)
class CampaignConfig:
    enabled: bool = False
    min_diameter_m: float = 0.09
    max_diameter_m: float = np.inf
    min_mu: float = 0.0  # areal density band [kg/m^2] of eligible objects
    max_mu: float = np.inf
    count: int = 0  # focus-set size; 0 = every eligible live object
    select: str = "random"  # random | lightest | heaviest (by mu) among eligible objects
    seed: int = 0
    exclusive: bool = True
    quality: float = 0.5  # non-exclusive: focus impulse >= quality x best candidate's impulse
    # every focus burst: predicted impulse per area >= min_impulse_frac x a full-length burst at the
    # Cm peak (phi_opt); otherwise wait (the platform retries each step as the range closes)
    min_impulse_frac: float = 0.15


def reference_impulse(fire_s: float, laser, params) -> float:
    """Predicted (expected-value) retrograde impulse per area [m/s x kg/m^2] of a burst of
    ``fire_s`` with every pulse at phi_opt, pushing along the LOS (as predict_bursts)."""
    return (fire_s * laser.prf_hz * params.eta1_nominal * np.cos(params.off_axis_mean_rad)
            * params.cm * params.phi_opt)


def select_focus(cfg: CampaignConfig, pop, dead: np.ndarray) -> np.ndarray:
    """Indices of the focus set among live objects of ``pop`` (sorted)."""
    if not cfg.enabled:
        return np.zeros(0, dtype=int)
    d, mu = pop.diameter_m, pop.areal_density
    ok = (~dead[:len(pop)] & (d >= cfg.min_diameter_m) & (d <= cfg.max_diameter_m)
          & (mu >= cfg.min_mu) & (mu <= cfg.max_mu))
    idx = np.flatnonzero(ok)
    n = len(idx) if cfg.count <= 0 else min(cfg.count, len(idx))
    if cfg.select == "random":
        idx = np.random.default_rng(cfg.seed).permutation(idx)[:n]
    elif cfg.select in ("lightest", "heaviest"):
        order = np.argsort(mu[idx], kind="stable")
        idx = idx[order[:n] if cfg.select == "lightest" else order[::-1][:n]]
    else:
        raise ValueError(f"campaign.select {cfg.select!r}: expected random, lightest or heaviest")
    return np.sort(idx)


def focus_need(pop, idx: np.ndarray) -> np.ndarray:
    """Ideal retrograde dv [m/s] still needed to bring each focus object to the removal perigee."""
    if len(idx) == 0:
        return np.zeros(0)
    el = pop.elements[idx]
    return dv_to_remove(el.a, el.e)


def focus_summary(idx: np.ndarray, need0: np.ndarray, need1: np.ndarray, removed: np.ndarray, pop,
                  events) -> dict:
    """Campaign metrics for SimResult.summary()."""
    if len(idx) == 0:
        return {}
    rem = removed[idx]
    prog = np.where(rem, 1.0, np.clip((need0 - need1) / np.maximum(need0, 1e-9), 0.0, 1.0))
    fset = set(idx.tolist())
    fev = [e for e in events if e.debris_index in fset]
    mu = pop.areal_density[idx]
    return {
        "focus_objects": int(len(idx)),
        "focus_removed": int(rem.sum()),
        "focus_bursts": len(fev),
        "focus_burst_frac": len(fev) / max(len(events), 1),
        "focus_objects_hit": len({e.debris_index for e in fev}),
        "focus_dv_total_m_s": float(sum(max(e.dv_retrograde, 0.0) for e in fev)),
        "focus_dv_per_burst": float(np.mean([e.dv_retrograde for e in fev])) if fev else 0.0,
        "focus_need_median_m_s": float(np.median(need0)),
        "focus_progress_mean": float(prog.mean()),
        "focus_progress_median": float(np.median(prog)),
        "focus_progress_max": float(prog.max()),
        "focus_half_way": int((prog >= 0.5).sum()),
        "focus_mu_median": float(np.median(mu)),
        "focus_removed_mu_max": float(mu[rem].max()) if rem.any() else 0.0,
    }
