// TALOS playback viewer. Data: #talos-data (talos.render.export_playback); propagation: TalosOrbit.
(function () {
  "use strict";
  const O = window.TalosOrbit;
  const D = JSON.parse(document.getElementById("talos-data").textContent);
  const meta = D.meta;
  const TYPES = { float64: Float64Array, float32: Float32Array, uint8: Uint8Array, int8: Int8Array, int32: Int32Array };
  const A = {};
  for (const [k, v] of Object.entries(D.arrays)) {
    const bin = atob(v.b64);
    const buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    A[k] = new TYPES[v.dtype](buf.buffer);
  }

  const N = meta.n_obj, E = meta.n_events, P = meta.platforms.length;
  const PCOL = ["#22d3ee", "#a78bfa", "#34d399", "#f472b6", "#facc15", "#60a5fa", "#fb923c", "#e879f9"];
  const COL = { untouched: "rgba(170,180,195,0.55)", engaged: "#fbbf24", removed: "#fb7185", target: "#ffffff" };

  // ---------- orbital state (viewer clock t = seconds since window start) ----------
  const st = { a: new Float64Array(N), e: new Float64Array(N), i: new Float64Array(N), raan: new Float64Array(N),
               argp: new Float64Array(N), M: new Float64Array(N), ep: new Float64Array(N) };
  const rates = new Float64Array(3 * N);
  const adot = new Float64Array(N), edot = new Float64Array(N); // drag: da/dt [m/s], de/dt [1/s]
  const engaged = new Uint8Array(N), removed = new Uint8Array(N), visible = new Uint8Array(N);
  const plat = { a: new Float64Array(P), e: new Float64Array(P), i: new Float64Array(P), raan: new Float64Array(P),
                 argp: new Float64Array(P), M: new Float64Array(P), ep: new Float64Array(P) };
  const platRates = new Float64Array(3 * P);
  const EL = ["a", "e", "i", "raan", "argp", "M"];
  for (let p = 0; p < P; p++) {
    EL.forEach((k, j) => (plat[k][p] = A.plat_el[6 * p + j]));
    O.j2Rates(plat.a[p], plat.e[p], plat.i[p], platRates, p);
  }
  // events per platform, in time order (export sorts by t_stop; bursts of one platform do not overlap)
  const byPlat = Array.from({ length: P }, () => []);
  for (let k = 0; k < E; k++) byPlat[A.ev_plat[k]].push(k);
  const evOf = new Map(); // object -> its window events
  for (let k = 0; k < E; k++) { const o = A.ev_obj[k]; if (!evOf.has(o)) evOf.set(o, []); evOf.get(o).push(k); }

  let applied = 0, tApplied = -Infinity, snapsDone = 0;
  const ptr = new Int32Array(P);
  const tally = { bursts: 0, removed: 0, dv: 0 };
  let colourDirty = true;
  // bursts are applied when they end; the export orders them by start
  const order = Array.from({ length: E }, (_, k) => k).sort((x, y) => A.ev_t1[x] - A.ev_t1[y]);
  const S = A.snap_t.length;
  // objects that enter (new background / replacement objects) or decay naturally in the window
  const changing = [];
  for (let o = 0; o < N; o++) if (A.obj_t_add[o] > 0 || A.obj_t_gone[o] < Infinity) changing.push(o);
  const liveAt = (o, t) => (A.obj_t_add[o] <= t && t < A.obj_t_gone[o] ? 1 : 0);

  function reset() {
    for (const k of EL) st[k].set(A["obj_" + k]);
    st.ep.set(A.obj_ep);
    adot.set(A.obj_adot); edot.set(A.obj_edot);
    for (let o = 0; o < N; o++) O.j2Rates(st.a[o], st.e[o], st.i[o], rates, o);
    engaged.set(A.obj_engaged_before); removed.set(A.obj_removed_before);
    for (let o = 0; o < N; o++) visible[o] = liveAt(o, 0);
    applied = 0; snapsDone = 0; tApplied = -Infinity; ptr.fill(0);
    tally.bursts = 0; tally.removed = 0; tally.dv = 0;
    colourDirty = true;
  }
  function applyEvent(k) {
    const o = A.ev_obj[k];
    if (A.ev_keep_el[k]) {  // else a decay step during the burst already holds the sim's elements
      EL.forEach((name, j) => (st[name][o] = A.ev_el[6 * k + j]));
      st.ep[o] = A.ev_tm[k];
      adot[o] = A.ev_rate[2 * k]; edot[o] = A.ev_rate[2 * k + 1];
      O.j2Rates(st.a[o], st.e[o], st.i[o], rates, o);
    }
    engaged[o] = 1;
    if (A.ev_removed[k] && !removed[o]) { removed[o] = 1; tally.removed++; }
    tally.bursts++; tally.dv += A.ev_dv[k];
    if (!colourDirty) styleObj(o);
  }
  // sparse decay steps: snapshot s updates objects snap_idx[off[s] .. off[s] + snap_n[s])
  const snapOff = new Int32Array(S + 1);
  for (let s = 0; s < S; s++) snapOff[s + 1] = snapOff[s] + A.snap_n[s];
  function applySnapshot(s) {  // the sim's decay step: run it locally, then apply corrections
    const u = A.snap_t[s];
    for (let o = 0; o < N; o++)
      if (!removed[o] && A.obj_t_add[o] <= u && u < A.obj_t_gone[o]) O.decayStep(st, rates, adot, edot, o, u);
    for (let j = snapOff[s]; j < snapOff[s + 1]; j++) {
      const o = A.snap_idx[j], q = 6 * j;
      st.a[o] = A.snap_el[q]; st.e[o] = A.snap_el[q + 1]; st.i[o] = A.snap_el[q + 2];
      st.raan[o] = A.snap_el[q + 3]; st.argp[o] = A.snap_el[q + 4]; st.M[o] = A.snap_el[q + 5];
      st.ep[o] = u;
      adot[o] = A.snap_rate[2 * j]; edot[o] = A.snap_rate[2 * j + 1];
      O.j2Rates(st.a[o], st.e[o], st.i[o], rates, o);
    }
  }
  function advanceTo(t) {
    if (t < tApplied) reset();
    for (;;) {  // snapshots and burst ends, in time order (snapshot first on ties)
      const ts = snapsDone < S ? A.snap_t[snapsDone] : Infinity;
      const te = applied < E ? A.ev_t1[order[applied]] : Infinity;
      if (ts <= te && ts <= t) applySnapshot(snapsDone++);
      else if (te <= t) applyEvent(order[applied++]);
      else break;
    }
    for (const o of changing) {
      const v = liveAt(o, t);
      if (v !== visible[o]) { visible[o] = v; if (!colourDirty) styleObj(o); }
    }
    for (let p = 0; p < P; p++) {
      const L = byPlat[p];
      while (ptr[p] < L.length && A.ev_t1[L[ptr[p]]] < t) ptr[p]++;
    }
    tApplied = t;
  }
  const activeEvent = (p, t) => { const L = byPlat[p], k = L[ptr[p]]; return k !== undefined && A.ev_t0[k] <= t ? k : -1; };
  const nextEvent = (p) => byPlat[p][ptr[p]];

  // ---------- Cesium ----------
  const C = window.Cesium;
  const viewer = new C.Viewer("cesium", {
    baseLayer: C.ImageryLayer.fromProviderAsync(
      C.TileMapServiceImageryProvider.fromUrl(C.buildModuleUrl("Assets/Textures/NaturalEarthII"))),
    baseLayerPicker: false, geocoder: false, homeButton: false, sceneModePicker: false,
    navigationHelpButton: false, infoBox: false, selectionIndicator: false, fullscreenButton: false,
    animation: true, timeline: true, shouldAnimate: true,
  });
  const scene = viewer.scene;
  scene.globe.enableLighting = true;
  scene.globe.showGroundAtmosphere = true;
  scene.screenSpaceCameraController.minimumZoomDistance = 2e5;
  const start = C.JulianDate.fromIso8601(meta.epoch_iso);
  const stop = C.JulianDate.addSeconds(start, meta.duration_s, new C.JulianDate());
  const clock = viewer.clock;
  clock.startTime = start.clone(); clock.stopTime = stop.clone(); clock.currentTime = start.clone();
  clock.clockRange = C.ClockRange.LOOP_STOP; clock.multiplier = 60;
  viewer.timeline.zoomTo(start, stop);

  const pts = scene.primitives.add(new C.PointPrimitiveCollection());
  const points = new Array(N);
  const cUntouched = C.Color.fromCssColorString(COL.untouched), cEng = C.Color.fromCssColorString(COL.engaged),
        cRem = C.Color.fromCssColorString(COL.removed);
  for (let o = 0; o < N; o++) points[o] = pts.add({ position: C.Cartesian3.ZERO, pixelSize: 2, color: cUntouched, id: { kind: "obj", o } });
  const hl = scene.primitives.add(new C.PointPrimitiveCollection());
  const targets = [];
  const satPts = scene.primitives.add(new C.PointPrimitiveCollection());
  const labels = scene.primitives.add(new C.LabelCollection());
  const lines = scene.primitives.add(new C.PolylineCollection());
  const sats = [], beams = [], rings = [];
  const RING_N = 181;
  for (let p = 0; p < P; p++) {
    const col = C.Color.fromCssColorString(PCOL[p % PCOL.length]);
    sats.push(satPts.add({ position: C.Cartesian3.ZERO, pixelSize: 9, color: col, outlineColor: C.Color.BLACK, outlineWidth: 1.5,
                           id: { kind: "sat", p } }));
    labels.add({ position: C.Cartesian3.ZERO, text: `${meta.platforms[p].name} · ${Math.round(meta.platforms[p].alt_km)} km`,
                 font: "12px system-ui, sans-serif", fillColor: col, pixelOffset: new C.Cartesian2(10, -10),
                 style: C.LabelStyle.FILL_AND_OUTLINE, outlineColor: C.Color.BLACK, outlineWidth: 3 });
    beams.push(lines.add({ positions: [C.Cartesian3.ZERO, C.Cartesian3.UNIT_X], width: 4, show: false,
      material: C.Material.fromType("PolylineGlow", { color: col, glowPower: 0.3 }) }));
    rings.push(lines.add({ positions: [C.Cartesian3.ZERO, C.Cartesian3.UNIT_X], width: 1.2, show: false,
      material: C.Material.fromType("Color", { color: col.withAlpha(0.55) }) }));
    targets.push(hl.add({ position: C.Cartesian3.ZERO, pixelSize: 7, color: C.Color.WHITE, outlineColor: col, outlineWidth: 2, show: false }));
  }
  const selRing = lines.add({ positions: [C.Cartesian3.ZERO, C.Cartesian3.UNIT_X], width: 1.8, show: false,
    material: C.Material.fromType("Color", { color: C.Color.fromCssColorString(COL.engaged).withAlpha(0.85) }) });
  const selPt = hl.add({ position: C.Cartesian3.ZERO, pixelSize: 10, color: C.Color.TRANSPARENT,
                         outlineColor: C.Color.fromCssColorString("#fef08a"), outlineWidth: 2, show: false });

  // ---------- options ----------
  const opt = { inertial: true, orbits: true, byMu: false, density: 1, follow: -1, sel: -1 };
  const muCol = (mu) => {
    if (!(mu > 0)) return cUntouched;
    const f = Math.min(1, Math.max(0, (Math.log10(mu) + 0.7) / 2.0)); // mu 0.2 -> 0, 20 -> 1
    return new C.Color(0.37 + (0.28 - 0.37) * f, 0.92 + (0.33 - 0.92) * f, 0.83 + (0.41 - 0.83) * f, 0.75 - 0.25 * f);
  };
  function styleObj(o) {
    const pt = points[o];
    const bg = !engaged[o] && !removed[o];
    const shown = visible[o] && (!bg || (opt.density > 0 && o % opt.density === 0));
    pt.show = !!shown;
    if (!shown) return;
    if (removed[o]) { pt.color = cRem; pt.pixelSize = 4; }
    else if (engaged[o]) { pt.color = cEng; pt.pixelSize = 3; }
    else { pt.color = opt.byMu ? muCol(A.obj_mu[o]) : cUntouched; pt.pixelSize = 2; }
  }
  function recolour() {
    for (let o = 0; o < N; o++) styleObj(o);
    colourDirty = false;
  }

  // ---------- per-frame update ----------
  const R = new Float64Array(3), ring = new Float64Array(3 * RING_N);
  const tmp = new C.Cartesian3(), M3 = new C.Matrix3();
  const fixedOf = (x, y, z, out) => {
    const m = M3; // column-major
    out.x = m[0] * x + m[3] * y + m[6] * z; out.y = m[1] * x + m[4] * y + m[7] * z; out.z = m[2] * x + m[5] * y + m[8] * z;
    return out;
  };
  const objFixed = (o, t, out) => { O.positionAt(st, rates, o, t, R, 0, adot); return fixedOf(R[0], R[1], R[2], out); };
  const satFixed = (p, t, out) => { O.positionAt(plat, platRates, p, t, R, 0); return fixedOf(R[0], R[1], R[2], out); };
  const ringPos = new Map(); // polyline -> its own Cartesian3 array
  function setRing(line, o, t) {
    O.orbitRing(st, rates, o, t, RING_N, ring, adot);
    let pos = ringPos.get(line);
    if (!pos) { pos = Array.from({ length: RING_N }, () => new C.Cartesian3()); ringPos.set(line, pos); }
    for (let j = 0; j < RING_N; j++) fixedOf(ring[3 * j], ring[3 * j + 1], ring[3 * j + 2], pos[j]);
    line.positions = pos;
    line.show = true;
  }
  let tNow = 0;
  scene.preUpdate.addEventListener((sc, time) => {
    const t = C.JulianDate.secondsDifference(time, start);
    tNow = t;
    C.Transforms.computeTemeToPseudoFixedMatrix(time, M3);
    advanceTo(t);
    if (colourDirty) recolour();
    for (let o = 0; o < N; o++) {
      const pt = points[o];
      if (pt.show) pt.position = objFixed(o, t, tmp);
    }
    for (let p = 0; p < P; p++) {
      const sp = satFixed(p, t, new C.Cartesian3());
      sats[p].position = sp;
      labels.get(p).position = sp;
      const k = activeEvent(p, t);
      if (k >= 0) {
        const o = A.ev_obj[k];
        const tp = objFixed(o, t, new C.Cartesian3());
        beams[p].positions = [sp, tp]; beams[p].show = true;
        targets[p].position = tp; targets[p].show = true;
        if (opt.orbits) setRing(rings[p], o, t); else rings[p].show = false;
      } else { beams[p].show = false; targets[p].show = false; rings[p].show = false; }
    }
    if (opt.sel >= 0) { setRing(selRing, opt.sel, t); selPt.position = objFixed(opt.sel, t, new C.Cartesian3()); selPt.show = true; }
    else { selRing.show = false; selPt.show = false; }
  });

  // inertial camera / follow a platform (keep the user's offset while the frame moves)
  scene.postUpdate.addEventListener((sc, time) => {
    if (sc.mode !== C.SceneMode.SCENE3D) return;
    const cam = viewer.camera;
    if (opt.follow >= 0) {
      const sp = satFixed(opt.follow, C.JulianDate.secondsDifference(time, start), new C.Cartesian3());
      const off = C.Cartesian3.clone(cam.position);
      cam.lookAtTransform(C.Transforms.eastNorthUpToFixedFrame(sp), off);
    } else if (opt.inertial) {
      const off = C.Cartesian3.clone(cam.position);
      cam.lookAtTransform(C.Matrix4.fromRotationTranslation(C.Transforms.computeTemeToPseudoFixedMatrix(time)), off);
    }
  });
  viewer.camera.setView({ destination: C.Cartesian3.fromDegrees(-40, 25, 2.6e7) });

  // ---------- HUD ----------
  const $ = (id) => document.getElementById(id);
  const fmt = (x, d = 1) => (x === null || x === undefined || !isFinite(x) ? "—" : x.toFixed(d));
  const fmtDur = (s) => (s < 60 ? `${Math.round(s)} s` : s < 3600 ? `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s` : `${(s / 3600).toFixed(1)} h`);
  const objLabel = (o) => `#${A.obj_id[o]}`;
  const sizeCm = (o) => (A.obj_diam[o] > 0 ? `${(A.obj_diam[o] * 100).toFixed(1)} cm` : "? cm");
  $("title").textContent = `TALOS · ${meta.title}`;
  $("subtitle").textContent = `${P} platforms · ${meta.n_obj.toLocaleString()} objects drawn (of ${meta.n_population.toLocaleString()}) · `
    + `day ${fmt(meta.t0_day, 1)}–${fmt(meta.t0_day + meta.duration_s / 86400, 1)} · ${E.toLocaleString()} bursts`;
  const SPEEDS = [1, 10, 60, 600, 3600];
  const sp = $("speeds");
  const playBtn = document.createElement("button");
  playBtn.onclick = () => { clock.shouldAnimate = !clock.shouldAnimate; };
  sp.appendChild(playBtn);
  const speedBtns = SPEEDS.map((s) => {
    const b = document.createElement("button");
    b.textContent = `${s}×`;
    b.onclick = () => { clock.multiplier = s; clock.shouldAnimate = true; };
    sp.appendChild(b);
    return b;
  });
  const platRows = meta.platforms.map((pm, p) => {
    const row = document.createElement("div");
    row.className = "plat";
    row.innerHTML = `<div class="dot" style="background:${PCOL[p % PCOL.length]}"></div><div><span class="pname">${pm.name}</span>
      <span style="color:var(--muted)"> · ${Math.round(pm.alt_km)} km</span></div><div class="pstate"></div>
      <div class="bar"><i style="background:${PCOL[p % PCOL.length]}"></i></div>`;
    row.title = "Click to follow this platform (click again to stop)";
    row.onclick = () => follow(opt.follow === p ? -1 : p);
    $("plats").appendChild(row);
    return { state: row.querySelector(".pstate"), bar: row.querySelector(".bar i"), row };
  });
  const legend = [[COL.untouched, "untouched debris (1–10 cm)"], [COL.engaged, "engaged at least once"],
    [COL.removed, "removed (lifetime < 30 d)"], ["#fff", "current target, beam in platform colour"]];
  $("legend").innerHTML = legend.map(([c, s]) => `<div class="sw" style="background:${c}"></div><div>${s}</div>`).join("")
    + (meta.engaged_before ? `<div></div><div>${meta.engaged_before.toLocaleString()} engaged and ${meta.removed_before.toLocaleString()} removed before this window</div>` : "");
  const mk = meta.miss_km;
  $("foot").innerHTML = `Propagation: J2 secular mean elements, no drag. Target position vs the sim at mid-burst: median ${fmt(mk.median, 2)} km, `
    + `95th pct ${fmt(mk.p95, 1)} km. Space: play/pause · click a point for details.`;

  function follow(p) {
    opt.follow = p;
    const cam = viewer.camera;
    platRows.forEach((r, q) => (r.row.style.background = q === p ? "rgba(148,163,184,.10)" : ""));
    if (p >= 0) {
      const sp0 = satFixed(p, tNow, new C.Cartesian3());
      cam.lookAtTransform(C.Transforms.eastNorthUpToFixedFrame(sp0), new C.HeadingPitchRange(0, -0.5, 2.5e6));
    } else {
      cam.lookAtTransform(C.Matrix4.IDENTITY);
      cam.setView({ destination: C.Cartesian3.fromDegrees(-40, 25, 2.6e7) });
    }
  }

  function renderInfo() {
    const box = $("info");
    const o = opt.sel;
    if (o < 0) { box.style.display = "none"; return; }
    const t = tNow;
    const a = st.a[o], e = st.e[o];
    const rp = (a * (1 - e) - O.RE) / 1e3, ra = (a * (1 + e) - O.RE) / 1e3;
    const mat = A.obj_mat[o] >= 0 ? meta.materials[A.obj_mat[o]] : "?";
    const firing = [];
    for (let p = 0; p < P; p++) { const k = activeEvent(p, t); if (k >= 0 && A.ev_obj[k] === o) firing.push(meta.platforms[p].name); }
    const status = (removed[o] ? "removed" : engaged[o] ? "engaged" : "not yet engaged")
      + (firing.length ? ` · being fired on by ${firing.join(", ")}` : "");
    const hist = (evOf.get(o) || []).map((k) => {
      const done = A.ev_t1[k] <= t;
      return `<tr style="opacity:${done ? 1 : 0.4}"><td>${fmt((A.ev_t0[k]) / 86400 + meta.t0_day, 2)} d · ${meta.platforms[A.ev_plat[k]].name}</td>`
        + `<td>${fmt(A.ev_dv[k], 1)} m/s · ${Math.round(A.ev_rp0[k])}→${Math.round(A.ev_rp1[k])} km${A.ev_removed[k] ? ' <span class="badge">REMOVED</span>' : ""}</td></tr>`;
    }).join("");
    box.innerHTML = `<button class="x" title="Close">×</button><h1>Object ${objLabel(o)}</h1>
      <div class="sub">${status}</div>
      <h2>Properties</h2><table>
      <tr><td>Size</td><td>${sizeCm(o)}</td></tr>
      <tr><td>Areal density μ</td><td>${fmt(A.obj_mu[o], 2)} kg/m²</td></tr>
      <tr><td>Material class</td><td>${mat}</td></tr>
      <tr><td>Perigee / apogee</td><td>${Math.round(rp)} / ${Math.round(ra)} km</td></tr>
      <tr><td>Inclination</td><td>${fmt((st.i[o] * 180) / Math.PI, 2)}°</td></tr>
      <tr><td>Eccentricity</td><td>${fmt(e, 4)}</td></tr></table>
      <h2>Bursts in this window</h2>${hist ? `<table>${hist}</table>` : '<div class="sub">none</div>'}`;
    box.querySelector(".x").onclick = () => select(-1);
    box.style.display = "block";
  }
  function select(o) { opt.sel = o; renderInfo(); }

  function renderHud() {
    const t = tNow;
    const d = C.JulianDate.toIso8601(clock.currentTime, 0).replace("T", " ").replace("Z", " UTC");
    $("clock").firstChild.nodeValue = `Day ${fmt(meta.t0_day + t / 86400, 3)}`;
    $("clock-utc").textContent = d;
    playBtn.textContent = clock.shouldAnimate ? "❚❚" : "▶";
    speedBtns.forEach((b, j) => b.classList.toggle("on", clock.multiplier === SPEEDS[j]));
    $("s-bursts").textContent = tally.bursts.toLocaleString();
    $("s-removed").textContent = tally.removed.toLocaleString();
    $("s-dv").textContent = Math.round(tally.dv).toLocaleString();
    for (let p = 0; p < P; p++) {
      const k = activeEvent(p, t), r = platRows[p];
      if (k >= 0) {
        const o = A.ev_obj[k];
        r.state.textContent = `firing → ${objLabel(o)} · ${sizeCm(o)} · μ ${fmt(A.obj_mu[o], 2)} · ${fmt(A.ev_dv[k], 1)} m/s`;
        r.bar.style.width = `${(100 * (t - A.ev_t0[k])) / (A.ev_t1[k] - A.ev_t0[k])}%`;
      } else {
        const nk = nextEvent(p);
        r.state.textContent = nk === undefined ? "idle · no more bursts in window" : `recovering · next burst in ${fmtDur(A.ev_t0[nk] - t)}`;
        r.bar.style.width = "0";
      }
    }
    const recent = [];
    for (let j = applied - 1; j >= 0 && recent.length < 7; j--) recent.push(order[j]);
    $("feed").innerHTML = recent.length ? recent.map((k) => `<div data-o="${A.ev_obj[k]}">${meta.platforms[A.ev_plat[k]].name} → ${objLabel(A.ev_obj[k])} `
      + `${fmt(A.ev_dv[k], 1)} m/s · ${Math.round(A.ev_rp0[k])}→${Math.round(A.ev_rp1[k])} km${A.ev_removed[k] ? ' <span class="badge">REMOVED</span>' : ""}</div>`).join("")
      : "<div>none yet</div>";
    if (opt.sel >= 0) renderInfo();
  }
  setInterval(renderHud, 250);
  $("feed").addEventListener("click", (ev) => { const el = ev.target.closest("[data-o]"); if (el) select(+el.dataset.o); });

  $("t-inertial").onchange = (ev) => {
    opt.inertial = ev.target.checked;
    if (!opt.inertial && opt.follow < 0) viewer.camera.lookAtTransform(C.Matrix4.IDENTITY);
  };
  $("t-orbits").onchange = (ev) => (opt.orbits = ev.target.checked);
  $("t-mu").onchange = (ev) => { opt.byMu = ev.target.checked; colourDirty = true; };
  $("t-density").onchange = (ev) => { opt.density = +ev.target.value; colourDirty = true; };
  $("t-find").onchange = (ev) => {
    const id = +ev.target.value;
    for (let o = 0; o < N; o++) if (A.obj_id[o] === id) { select(o); return; }
    ev.target.style.borderColor = "var(--rose)";
    setTimeout(() => (ev.target.style.borderColor = ""), 1200);
  };
  document.addEventListener("keydown", (ev) => {
    if (ev.code === "Space" && ev.target.tagName !== "INPUT") { clock.shouldAnimate = !clock.shouldAnimate; ev.preventDefault(); }
  });
  const handler = new C.ScreenSpaceEventHandler(scene.canvas);
  handler.setInputAction((click) => {
    const picked = scene.pick(click.position, 9, 9);
    const id = picked && picked.id;
    if (id && id.kind === "obj") select(id.o);
    else if (id && id.kind === "sat") follow(opt.follow === id.p ? -1 : id.p);
  }, C.ScreenSpaceEventType.LEFT_CLICK);

  reset();
  document.getElementById("loading").remove();
  if (location.protocol === "file:") {
    const b = $("filewarn");
    b.style.display = "block";
    $("filewarn-cmd").textContent = `python -m http.server 8765 -d "${decodeURIComponent(location.pathname.replace(/^\/([A-Za-z]:)/, "$1").replace(/\/[^/]*$/, ""))}"`;
  }
  window.talos = { viewer, clock, select, follow, opt, meta, A, events: { t0: A.ev_t0, t1: A.ev_t1, plat: A.ev_plat },
                   setTime: (s) => { clock.currentTime = C.JulianDate.addSeconds(start, s, new C.JulianDate()); } };
})();
