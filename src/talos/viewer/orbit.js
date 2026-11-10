// Mean-element propagation for the TALOS playback viewer: a port of
// talos.propagator.j2_secular_rates + talos.elements.elements_to_rv (J2 secular,
// no drag). Positions are in the sim's inertial frame (mean equator, ~J2000), metres.
// Works in the browser (window.TalosOrbit) and in Node (module.exports) for tests.
(function (root) {
  "use strict";
  const MU = 3.986004418e14;
  const RE = 6378137.0;
  const J2 = 1.08262668e-3;
  const TWO_PI = 2 * Math.PI;

  // Secular rates [rad/s] for one element set; writes into out[k*3 .. k*3+2].
  function j2Rates(a, e, i, out, k) {
    const n = Math.sqrt(MU / (a * a * a));
    const p = a * (1 - e * e);
    const kk = 0.75 * n * J2 * (RE / p) * (RE / p);
    const ci = Math.cos(i);
    out[3 * k] = -2 * kk * ci; // raan_dot
    out[3 * k + 1] = kk * (5 * ci * ci - 1); // argp_dot
    out[3 * k + 2] = n + kk * Math.sqrt(1 - e * e) * (3 * ci * ci - 1); // M_dot
  }

  function solveKepler(M, e) {
    M = M % TWO_PI;
    if (M < 0) M += TWO_PI;
    let E = e < 0.8 ? M : Math.PI;
    for (let it = 0; it < 30; it++) {
      const dE = -(E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
      E += dE;
      if (Math.abs(dE) < 1e-12) break;
    }
    return E;
  }

  // Inertial position of element set k at time t (seconds, same clock as ep[k]).
  // el: {a, e, i, raan, argp, M, ep} typed arrays; rates: Float64Array(3n) from j2Rates.
  function positionAt(el, rates, k, t, out, o) {
    const dt = t - el.ep[k];
    const a = el.a[k], e = el.e[k], inc = el.i[k];
    const raan = el.raan[k] + rates[3 * k] * dt;
    const argp = el.argp[k] + rates[3 * k + 1] * dt;
    const M = el.M[k] + rates[3 * k + 2] * dt;
    const E = solveKepler(M, e);
    const x = a * (Math.cos(E) - e);
    const y = a * Math.sqrt(1 - e * e) * Math.sin(E);
    const cO = Math.cos(raan), sO = Math.sin(raan);
    const cw = Math.cos(argp), sw = Math.sin(argp);
    const ci = Math.cos(inc), si = Math.sin(inc);
    out[o] = x * (cO * cw - sO * sw * ci) + y * (-cO * sw - sO * cw * ci);
    out[o + 1] = x * (sO * cw + cO * sw * ci) + y * (-sO * sw + cO * cw * ci);
    out[o + 2] = x * (sw * si) + y * (cw * si);
  }

  // n points around the orbit of element set k, frozen at time t (for drawing the ellipse).
  function orbitRing(el, rates, k, t, n, out) {
    const dt = t - el.ep[k];
    const a = el.a[k], e = el.e[k], inc = el.i[k];
    const raan = el.raan[k] + rates[3 * k] * dt;
    const argp = el.argp[k] + rates[3 * k + 1] * dt;
    const cO = Math.cos(raan), sO = Math.sin(raan);
    const cw = Math.cos(argp), sw = Math.sin(argp);
    const ci = Math.cos(inc), si = Math.sin(inc);
    for (let j = 0; j < n; j++) {
      const E = (TWO_PI * j) / (n - 1);
      const x = a * (Math.cos(E) - e);
      const y = a * Math.sqrt(1 - e * e) * Math.sin(E);
      out[3 * j] = x * (cO * cw - sO * sw * ci) + y * (-cO * sw - sO * cw * ci);
      out[3 * j + 1] = x * (sO * cw + cO * sw * ci) + y * (-sO * sw + cO * cw * ci);
      out[3 * j + 2] = x * (sw * si) + y * (cw * si);
    }
  }

  const api = { MU, RE, J2, j2Rates, solveKepler, positionAt, orbitRing };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TalosOrbit = api;
})(typeof window !== "undefined" ? window : this);
