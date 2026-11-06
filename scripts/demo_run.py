"""First end-to-end run on the TOY population (not MASTER) — plumbing check, not a result.

usage: python scripts/demo_run.py [n_samples] [n_sats] [hours] [mu]
"""

import sys
import time

import numpy as np

from talos.metrics import raw_encounter_rate, required_removal_rate
from talos.platform import baseline_constellation
from talos.population import toy_population
from talos.simulation import run_engagements

n, sats, hours, mu = (int(sys.argv[1]) if len(sys.argv) > 1 else 5000,
                      int(sys.argv[2]) if len(sys.argv) > 2 else 4,
                      float(sys.argv[3]) if len(sys.argv) > 3 else 2.0,
                      float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
rng = np.random.default_rng(0)
pop = toy_population(n, rng, n_real=1.0e6, areal_density=mu)
plats = baseline_constellation(sats)

t0 = time.perf_counter()
enc = raw_encounter_rate(plats[0], pop, np.arange(0.0, 1800.0))
print(f"raw encounters <=325 km: {enc:.1f}/min per sat (gate 4 target ~25/min)  [{time.perf_counter()-t0:.1f}s]")

t0 = time.perf_counter()
res = run_engagements(plats, pop, hours * 3600.0, rng=np.random.default_rng(1))
print(f"sim {hours} h x {sats} sats x {n} samples: {time.perf_counter()-t0:.1f}s")
for k, v in res.summary().items():
    print(f"  {k:42s} {v:,.3f}" if isinstance(v, float) else f"  {k:42s} {v}")
print(f"  {'required removals/yr (90% in 10 yr)':42s} {required_removal_rate(1e6, 10):,.0f}")
