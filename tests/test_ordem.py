"""ORDEM interface: forward kernel, fit recovery, sampling, config wiring (no ORDEM install needed)."""

import numpy as np
import pytest

from talos import ordem
from talos.config import build, load_scenario
from talos.constants import MU_EARTH, R_EARTH, YEAR
from talos.ordem import MATERIALS, OrdemTable, fit_table, inclination_kernel, reduce_igloo, sample_cells

INC = np.arange(0.0, 180.1, 5.0)
VEL = np.arange(0.0, 24.0, 1.0)
Z = np.cos(np.deg2rad(INC[:-1])) - np.cos(np.deg2rad(INC[1:]))


@pytest.mark.parametrize("sc_inc", [0.0, 51.6, 98.6, 150.0])
def test_kernel_isotropic_flux_is_analytic(sc_inc):
    """Isotropic circular debris: flux = n * <|v_rel|> with n = N / (4 pi r^2) and <|v_rel|> = 4v/pi."""
    r = R_EARTH + 800e3
    v = np.sqrt(MU_EARTH / r)
    k = inclination_kernel(800.0, sc_inc, INC, VEL)
    assert (k @ (Z / 2)).sum() == pytest.approx(4 * v / np.pi / (4 * np.pi * r**2) * YEAR, rel=1e-6)


def test_kernel_coplanar_counter_rotating_hits_at_2v():
    k = inclination_kernel(800.0, 0.0, INC, VEL)
    v2 = 2 * np.sqrt(MU_EARTH / (R_EARTH + 800e3)) / 1e3
    assert np.argmax(k[:, -1]) == int(v2)  # retrograde-equatorial debris vs prograde-equatorial sc
    assert k[:3, 0].sum() / k[:, 0].sum() > 0.9  # co-moving debris: low relative speed


def _synthetic_raw(true_c, alts=(780.0, 820.0), incs=(0.0, 65.0, 98.6, 140.0)):
    """Spectra the forward model predicts for per-metre counts ``true_c`` (same for all mats/sizes)."""
    flux = np.zeros((len(alts), len(incs), len(MATERIALS), 2, len(VEL) - 1))
    for a, h in enumerate(alts):
        for s, i in enumerate(incs):
            spec = inclination_kernel(h, i, INC, VEL) @ true_c
            flux[a, s] = spec * np.array([1.0, 0.25])[None, :, None]  # 3.16-10 cm = 1/4 of 1-3.16 cm
    return {"year": 2026, "alt_km": np.array(alts), "sc_inc_deg": np.array(incs), "v_edges_km_s": VEL,
            "flux": flux}


def test_fit_recovers_inclination_mix():
    true_c = np.zeros(len(INC) - 1)
    true_c[19] = 3e-4  # 95-100 deg (SSO)
    true_c[13] = 1e-4  # 65-70 deg
    true_c[16] = 5e-5  # 80-85 deg
    t = fit_table(_synthetic_raw(true_c), smooth=1e-4)
    got = t.count[0, 2, 0] / ((t.alt_edges_km[1] - t.alt_edges_km[0]) * 1e3)
    assert got.sum() == pytest.approx(true_c.sum(), rel=0.03)
    for j in (19, 13, 16):  # mass lands in (or next to) the right bins
        assert got[j - 1:j + 2].sum() == pytest.approx(true_c[j], rel=0.15)
    assert t.fit_residual[0, 2, 0] < 0.05
    assert t.count[0, 2, 1].sum() == pytest.approx(0.25 * t.count[0, 2, 0].sum(), rel=1e-6)


def test_reduce_igloo_differences_cumulative_sizes():
    cols = [m + s for m in MATERIALS for s in ("35", "40", "45", "50")]
    flux = np.zeros((2, len(cols)))
    k = cols.index
    flux[0, [k("MD40"), k("MD45"), k("MD50")]] = [10.0, 3.0, 1.0]
    flux[1, [k("MD40"), k("MD45"), k("MD50")]] = [4.0, 1.0, 0.0]
    out = reduce_igloo(np.array([7.0, 14.0]), np.array([8.0, 15.0]), cols, flux, VEL)
    md = MATERIALS.index("MD")
    assert out[md, 0, 7] == 7.0 and out[md, 1, 7] == 2.0
    assert out[md, 0, 14] == 3.0 and out[md, 1, 14] == 1.0
    assert out.sum() == 13.0


def _table(year=2026):
    alt_edges = np.array([590.0, 800.0, 1010.0])
    count = np.zeros((2, len(MATERIALS), 2, len(INC) - 1))
    count[0, 2, 0, 19] = 6000.0
    count[1, 2, 1, 13] = 2000.0
    count[1, 0, 0, 13] = 2000.0
    slope = np.full((2, len(MATERIALS), 2), 2.5)
    return OrdemTable(year, alt_edges, INC, count, slope, np.zeros((2, len(MATERIALS), 2)))


def test_sampling_matches_table(rng):
    t = _table()
    s = sample_cells(t, 40000, rng, alt_km=(600.0, 1000.0))
    assert s["represented"] == pytest.approx(t.total((600.0, 1000.0)))
    assert t.total((600.0, 1000.0)) == pytest.approx((6000 + 4000) * 200 / 210)  # both cells clipped 210 -> 200 km
    assert np.all((s["alt_km"] >= 600) & (s["alt_km"] <= 1000))
    assert np.all((s["diameter_m"] >= 0.01) & (s["diameter_m"] <= 0.1))
    i = np.rad2deg(s["inc"])
    assert np.mean((i > 95) & (i < 100)) == pytest.approx(6000 / 10000, abs=0.01)
    big = s["diameter_m"] > 10**-1.5
    assert np.all(s["material"][big] == 2) and np.mean(big) == pytest.approx(0.2, abs=0.01)
    small = s["diameter_m"][~big]  # N(>d) ~ d^-2.5: median of the 1-3.16 cm bin
    med = (0.5 * (0.01**-2.5 + 10**(-1.5 * 2.5))) ** (-1 / 2.5)
    assert np.median(small) == pytest.approx(med, rel=0.03)
    nk = sample_cells(t, 1000, rng, materials=("NK",))
    assert np.all(nk["material"] == 0)


def test_build_uses_ordem_table(tmp_path, monkeypatch):
    monkeypatch.setattr(ordem, "DATA_DIR", tmp_path)
    _table(2026).save()
    sc = load_scenario(overrides={"population.kind": "ordem", "population.n_samples": 3000,
                                  "population.min_lifetime_yr": None, "population.areal_density_model": "sphere",
                                  "constellation.n_sats": 1})
    b = build(sc)
    pop = b.population
    assert len(pop) == 3000
    assert pop.represented_count == pytest.approx(_table().total((600.0, 1000.0)))
    md = pop.areal_density[np.isclose(pop.areal_density / pop.diameter_m, 2 / 3 * 2800)]
    assert len(md) > 0.6 * len(pop)  # sphere model: mu = (2/3) rho d
    assert b.dynamics.weight == pytest.approx(pop.represented_count / 3000)
    new = b.dynamics.source(5, np.random.default_rng(0), 86400.0, b.dynamics.weight)
    assert len(new) == 5 and np.all(new.weight == b.dynamics.weight) and np.all(new.epoch_s == 86400.0)


def test_sbm_area_to_mass_matches_johnson_2001(rng):
    from talos.population import sbm_area_to_mass, sbm_log_am_params

    m, s = sbm_log_am_params(np.array([0.01, 10**-1.5, 0.1]))
    np.testing.assert_allclose(m, [-0.3, -0.65, -1.0])
    np.testing.assert_allclose(s, [0.2 + 0.1333 * 1.5, 0.2 + 0.1333 * 2.0, 0.2 + 0.1333 * 2.5])
    x = np.log10(sbm_area_to_mass(np.full(50000, 0.01), rng))
    assert np.mean(x) == pytest.approx(-0.3, abs=0.01) and np.std(x) == pytest.approx(0.4, rel=0.03)
    assert x.min() >= -0.3 - 3 * 0.4 - 1e-9


def test_sbm_model_keeps_nak_as_spheres(tmp_path, monkeypatch):
    monkeypatch.setattr(ordem, "DATA_DIR", tmp_path)
    _table(2026).save()
    sc = load_scenario(overrides={"population.kind": "ordem", "population.n_samples": 4000,
                                  "population.min_lifetime_yr": None, "population.areal_density_model": "sbm",
                                  "constellation.n_sats": 1})
    pop = build(sc).population
    sphere = np.isclose(pop.areal_density, 2 / 3 * 870.0 * pop.diameter_m)
    assert np.mean(sphere) == pytest.approx(2000 / 10000, abs=0.02)  # the NK cell
    frag = pop.areal_density[~sphere]
    assert 1.0 < np.median(frag) < 12.0


def test_missing_table_has_build_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(ordem, "DATA_DIR", tmp_path)
    sc = load_scenario(overrides={"population.kind": "ordem", "population.n_samples": 10})
    with pytest.raises(FileNotFoundError, match="build_ordem_table"):
        build(sc)


def test_toy_population_honours_sbm():
    sc = load_scenario(overrides={"population.kind": "toy", "population.n_real": 3000,
                                  "population.min_lifetime_yr": None, "population.areal_density_model": "sbm",
                                  "constellation.n_sats": 1})
    pop = build(sc).population
    assert np.std(pop.areal_density) > 0 and 1.0 < np.median(pop.areal_density) < 12.0
    fixed = build(load_scenario(overrides={"population.kind": "toy", "population.n_real": 3000,
                                           "population.min_lifetime_yr": None,
                                           "population.areal_density_model": "fixed",
                                           "constellation.n_sats": 1})).population
    np.testing.assert_array_equal(pop.diameter_m, fixed.diameter_m)  # same toy draws, only mu differs
    assert np.all(fixed.areal_density == 1.0)


def test_hanada_mixture_matches_olivieri_eqs(rng):
    from talos.population import hanada_area_to_mass, hanada_log_am_params

    a, mh, sh, ml, sl = hanada_log_am_params(np.array([0.01, 0.2]))
    # lambda = -2: alpha = 1 + 0.888 * 0.71 * -0.99; outside [-2.71, -1.01] high density only
    np.testing.assert_allclose(a, [1 + 0.888 * 0.71 * -0.99, 1.0])
    np.testing.assert_allclose([mh[0], sh[0], ml[0], sl[0]], [-0.264, 0.249, 0.344, 0.0762])
    x = np.log10(hanada_area_to_mass(np.full(50000, 0.01), rng))
    from math import erf, sqrt

    light = x > 0.1  # the low-density peak sits at +0.34 with sigma 0.08
    tail = lambda m, s: 0.5 * (1 - erf((0.1 - m) / (s * sqrt(2))))  # noqa: E731
    assert np.mean(light) == pytest.approx(a[0] * tail(mh[0], sh[0]) + (1 - a[0]) * tail(ml[0], sl[0]), abs=0.01)
    assert np.mean(x[light]) == pytest.approx(0.344, abs=0.01)
    y = np.log10(hanada_area_to_mass(np.full(50000, 0.2), rng))  # outside the fit: high density only
    assert np.mean(y) == pytest.approx(mh[1], abs=0.01) and np.std(y) == pytest.approx(sh[1], rel=0.03)


def test_hanada_and_proof_albedo_wiring(tmp_path, monkeypatch):
    monkeypatch.setattr(ordem, "DATA_DIR", tmp_path)
    _table(2026).save()
    base = {"population.kind": "ordem", "population.n_samples": 4000, "population.min_lifetime_yr": None,
            "constellation.n_sats": 1}
    sbm = build(load_scenario(overrides={**base, "population.areal_density_model": "sbm"})).population
    han = build(load_scenario(overrides={**base, "population.areal_density_model": "hanada",
                                         "population.albedo_model": "proof"})).population
    np.testing.assert_array_equal(sbm.diameter_m, han.diameter_m)
    nak = np.isclose(han.areal_density, 2 / 3 * 870.0 * han.diameter_m)
    assert np.median(han.areal_density[~nak]) < np.median(sbm.areal_density[~nak])
    assert np.median(han.albedo[nak]) == pytest.approx(0.85, rel=0.1)
    assert np.median(han.albedo[~nak]) == pytest.approx(0.2, rel=0.1)
    with pytest.raises(ValueError, match="albedo_model"):
        build(load_scenario(overrides={"population.kind": "toy", "population.n_real": 100,
                                       "population.albedo_model": "proof", "constellation.n_sats": 1}))


def test_hanada_material_uses_the_ordem_class(rng):
    from talos.population import hanada_area_to_mass, hanada_log_am_params

    d = np.full(20000, 0.01)
    ld = np.arange(20000) < 5000
    x = np.log10(hanada_area_to_mass(d, rng, low_density=ld))
    _, mh, sh, ml, sl = hanada_log_am_params(0.01)
    assert np.mean(x[ld]) == pytest.approx(ml, abs=0.01) and np.std(x[ld]) == pytest.approx(sl, rel=0.05)
    assert np.mean(x[~ld]) == pytest.approx(mh, abs=0.01) and np.std(x[~ld]) == pytest.approx(sh, rel=0.05)
