"""Receding-horizon commitment planner (module 6, second stage).

Why not a full-horizon plan: each burst (~10 m/s) shifts an object's period by
~0.4 %, i.e. ~2,600 km of along-track phase per day, so any pass predicted
beyond the next burst on that object is wrong. Only the *next* passes after
its latest burst are predictable. The planner therefore works one step ahead
per object:

  * Commit set: up to ``max_committed`` partly lowered objects, ranked by
    estimated bursts still needed (remaining dv-to-removal / mean realized dv).
  * Look-ahead: after every engagement on a committed object, re-screen that
    one object over ``horizon_s`` for all platforms to get its next real
    opportunities (with a predicted burst for each).
  * Reservation: a platform may fire a filler burst only if it will have
    recovered (true resource model) in time to take the committed object's
    next opportunity on that platform; otherwise it waits for it.
  * Committed candidates that are open now always win over fillers.

Objects with no viable opportunity in the horizon are released.

TODO: true multi-object optimisation over the predictable next step
(CP-SAT / MILP over reservations), cross-platform hand-off scoring.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .debris import DebrisPopulation
from .engagement import AblationParams
from .platform import Platform
from .resources import ResourceState
from .scheduler import Candidate, dv_to_remove, evaluate_candidates
from .screening import screen_windows


@dataclass(frozen=True)
class Opportunity:
    platform: int
    t_open: float
    t_close: float
    latest_select: float  # last grid time a burst can be chosen and still meet min_burst
    dv_pred: float


@dataclass
class _ObjState:
    dv_sum: float = 0.0
    hits: int = 0
    opps: list[Opportunity] = field(default_factory=list)
    ptr: int = 0  # first opportunity not yet expired


class CommitPlanner:
    def __init__(
        self,
        platforms: list[Platform],
        params: AblationParams,
        max_committed: int = 200,
        horizon_s: float = 6 * 3600.0,
        dt: float = 1.0,
        min_dv_frac: float = 0.25,
    ) -> None:
        self.platforms = platforms
        self.params = params
        self.max_committed = max_committed
        self.horizon_s = horizon_s
        self.dt = dt
        self.min_dv_frac = min_dv_frac  # opp must predict >= this x object's mean realized dv
        self.objs: dict[int, _ObjState] = {}
        self.committed: set[int] = set()
        self.retry_after: dict[int, float] = {}  # back-off for objects with no viable opportunity
        self.max_attempts_factor = 2  # commit attempts per refresh, x free slots
        self._cache_t0: float | None = None
        self._cache: list[tuple[np.ndarray, np.ndarray]] = []
        self._cache_len = 0
        self.stats = {"rescreens": 0, "released": 0, "blocked_fillers": 0, "committed_fired": 0}

    # ---- bookkeeping -------------------------------------------------------
    def is_committed(self, j: int) -> bool:
        return j in self.committed

    def record(self, j: int, dv_retro: float, t_now: float, pop: DebrisPopulation, removed: np.ndarray) -> None:
        """Call after every engagement (committed or filler)."""
        s = self.objs.setdefault(j, _ObjState())
        s.dv_sum += max(dv_retro, 0.0)
        s.hits += 1
        if j in self.committed:
            self.stats["committed_fired"] += 1
            if removed[j]:
                self.committed.discard(j)
            else:
                self._rescreen(j, t_now, pop)
                if not s.opps:
                    self._release(j, t_now)

    def refresh(self, t_now: float, pop: DebrisPopulation, removed: np.ndarray) -> None:
        """Drop removed/stranded objects and top up the commit set."""
        for j in list(self.committed):
            if removed[j]:
                self.committed.discard(j)
                continue
            s = self.objs[j]
            self._advance(s, t_now)
            if s.ptr >= len(s.opps):
                self._rescreen(j, t_now, pop)
                if not s.opps:
                    self._release(j, t_now)
        free = self.max_committed - len(self.committed)
        if free <= 0:
            return
        pool = [j for j, s in self.objs.items()
                if j not in self.committed and not removed[j] and s.dv_sum > 0
                and self.retry_after.get(j, -np.inf) <= t_now]
        if not pool:
            return
        pool_arr = np.array(pool)
        el = pop.elements[pool_arr]
        need = dv_to_remove(el.a, el.e)
        mean_dv = np.array([self.objs[j].dv_sum / self.objs[j].hits for j in pool])
        bursts_left = need / np.maximum(mean_dv, 1e-6)
        attempts = self.max_attempts_factor * free
        for i in np.argsort(bursts_left)[:attempts]:
            if free <= 0:
                break
            j = int(pool_arr[i])
            self._rescreen(j, t_now, pop)
            if self.objs[j].opps:
                self.committed.add(j)
                free -= 1
            else:
                self.retry_after[j] = t_now + 0.5 * self.horizon_s

    def _release(self, j: int, t_now: float) -> None:
        self.committed.discard(j)
        self.retry_after[j] = t_now + 0.5 * self.horizon_s
        self.stats["released"] += 1

    def _sat_states(self, t_from: float, n: int):
        """Cached platform states on an integer-``dt`` grid covering [t_from, t_from + n*dt).

        Returns (aligned start time, cache list of (r, v) per platform, start index).
        """
        from .propagator import states_at

        end_needed = t_from + (n + 1) * self.dt
        stale = (self._cache_t0 is None or t_from < self._cache_t0
                 or end_needed > self._cache_t0 + self._cache_len * self.dt)
        if stale:
            t0 = np.floor(t_from / self.dt) * self.dt
            m = n + int(2 * self.horizon_s / self.dt) + 2
            tt = t0 + np.arange(m) * self.dt
            self._cache = [states_at(p.orbit, tt) for p in self.platforms]
            self._cache_t0, self._cache_len = t0, m
        i0 = int(np.ceil((t_from - self._cache_t0) / self.dt - 1e-9))
        return self._cache_t0 + i0 * self.dt, self._cache, i0

    @staticmethod
    def _advance(s: _ObjState, t_now: float) -> None:
        while s.ptr < len(s.opps) and s.opps[s.ptr].latest_select < t_now:
            s.ptr += 1

    def _rescreen(self, j: int, t_from: float, pop: DebrisPopulation) -> None:
        """Next viable opportunities for object j on every platform over the horizon."""
        self.stats["rescreens"] += 1
        s = self.objs.setdefault(j, _ObjState())
        n = int(round(self.horizon_s / self.dt))
        t_start, cache, i0 = self._sat_states(t_from, n)
        t = t_start + np.arange(n) * self.dt
        mean_dv = s.dv_sum / s.hits if s.hits else 0.0
        cand = np.array([j])
        opps: list[Opportunity] = []
        for p_idx, plat in enumerate(self.platforms):
            res, atp = plat.resources, plat.atp
            r_c, v_c = cache[p_idx]
            states = (r_c[i0:i0 + n], v_c[i0:i0 + n])
            for w in screen_windows(plat, pop, t, candidates=cand, sat_states=states):
                c = evaluate_candidates(plat, pop, [w], t, w.start, self.params)
                if not c or c[0].dv_retrograde < self.min_dv_frac * mean_dv:
                    continue
                t_close = t[w.stop - 1] + self.dt
                opps.append(Opportunity(p_idx, t[w.start], t_close,
                                        t_close - res.min_burst_s - atp.settle_s, c[0].dv_retrograde))
        opps.sort(key=lambda o: o.t_open)
        s.opps, s.ptr = opps, 0

    # ---- reservation check -------------------------------------------------
    def next_reservation(self, p_idx: int, t_now: float) -> tuple[Opportunity, int] | None:
        """Earliest still-feasible next opportunity of any committed object on platform p."""
        best = None
        for j in self.committed:
            s = self.objs[j]
            self._advance(s, t_now)
            if s.ptr < len(s.opps):
                o = s.opps[s.ptr]
                if o.platform == p_idx and (best is None or o.latest_select < best[0].latest_select):
                    best = (o, j)
        return best

    def filler_blocks(self, p_idx: int, rs: ResourceState, c: Candidate, plat: Platform) -> tuple[bool, float]:
        """Would firing filler ``c`` stop platform p taking its next reservation?

        Returns (blocked, time to resume looking if blocked).
        """
        nxt = self.next_reservation(p_idx, c.t_start - plat.atp.settle_s)
        if nxt is None:
            return False, 0.0
        o, _ = nxt
        sim = copy.copy(rs)
        sim.advance(max(sim.t, c.t_start))
        sim.fire(c.t_stop - c.t_start)
        f_res = min(plat.resources.burst_s, max(o.t_close - o.t_open, plat.resources.min_burst_s))
        ready = sim.t + sim.wait_for(f_res)
        if ready <= o.latest_select:
            return False, 0.0
        self.stats["blocked_fillers"] += 1
        return True, max(o.t_open - plat.atp.settle_s, c.t_start - plat.atp.settle_s + self.dt)
