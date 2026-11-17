// robotd dashboard: one joint in full: the hover card, the joint tabs on the right (beside the log), and their charts.
"use strict";

window.Joints = (() => {
  const HISTORY_S = 30;
  const GUARD_TRACK_DEG = 0.15 * 180 / Math.PI;     // the driver's guard: commanded within 0.15 rad of measured
  const GUARD_EFFORT = 0.8;                         // and an arm joint above 80 % of its motor limit for 40 ms faults
  const tip = $("jtip");
  let live = null, meta = null, hovered = null, mouse = [0, 0], lastPaint = 0;
  const tabs = new Map();                           // "left:2" -> {arm, k, btn, pane}
  let active = "log";
  const hist = [];                                  // per live frame: {t, q, qr, dq, eff, temps}

  // ---------------------------------------------------------------- the force on a gripper (k = "tcp")
  function tcpInfo(arm) {
    const e = live.ee?.[arm] || { wrench: [0, 0, 0, 0, 0, 0], T: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]] };
    const F = e.wrench.slice(0, 3).map((v) => -v), M = e.wrench.slice(3, 6).map((v) => -v);   // on the tool
    const n = Math.hypot(...F);
    return { arm, k: "tcp", tcp: true, label: `${arm} tcp`, role: "force on the gripper", F, M, n, mn: Math.hypot(...M),
             dir: n > 1e-6 ? F.map((v) => v / n) : [0, 0, 0], p: [e.T[0][3], e.T[1][3], e.T[2][3]],
             contact: e.contact, passive: meta.passive };
  }

  function tcpRows(j) {
    const r = [];
    const add = (k, v) => r.push(`<span>${k}</span><span>${v}</span>`);
    if (j.passive) { add("force", "— (motors off: no estimate)"); return r.join(""); }
    add("force", `${n(j.n, 2)} N${j.contact ? " · contact" : ""}`);
    add("vector (world)", `x ${n(j.F[0], 2)}  y ${n(j.F[1], 2)}  z ${n(j.F[2], 2)} N`);
    add("direction", `(${j.dir.map((v) => n(v, 2)).join(", ")})`);
    add("torque about the tip", `${n(j.mn, 3)} Nm (${j.M.map((v) => n(v, 2)).join(", ")})`);
    add("tip at", `${j.p.map((v) => n(v * 1000, 0)).join(", ")} mm`);
    add("from", "all six joints: (model feed-forward − measured torque) through the Jacobian, ~25 ms filter, minus the zero");
    return r.join("");
  }

  // ---------------------------------------------------------------- everything known about one joint, now
  function info(arm, k) {
    if (k === "tcp") return tcpInfo(arm);
    const finger = k === 6;
    const name = finger ? meta.finger_joint[arm] : meta.arm_joints[arm][k];
    const i = meta.joints.indexOf(name);
    const h = live.health?.backend?.arms?.[arm] || {};
    const lim = meta.limits?.[i];
    const soft = !!(live.fault && live.fault.soft);
    const toDisp = (v) => (v === null || v === undefined) ? null : (finger ? 2000 * v : deg(v));   // mm of opening, or deg
    const q = live.q[i], qr = live.q_ref[i], qc = live.q_cmd?.[i];
    const eff = live.effort[i], ext = live.effort_ext?.[i], effLim = meta.effort_limit[i];
    const motor = h.motors?.[k];
    const group = finger ? `${arm}_gripper` : arm;
    return {
      arm, k, i, name, finger, label: finger ? `${arm} gripper` : `${arm} J${k + 1}`, role: finger ? "gripper" : ROLE[k],
      unit: finger ? "mm" : "°", q, qr, qc, dq: live.dq[i],
      pos: toDisp(q), ref: toDisp(qr), cmd: toDisp(qc), err: toDisp(qr - q), vel: finger ? 2000 * live.dq[i] : deg(live.dq[i]),
      lo: lim ? toDisp(lim[0]) : null, hi: lim ? toDisp(lim[1]) : null,
      frac: lim ? (q - lim[0]) / (lim[1] - lim[0]) : null,
      toLimit: lim ? Math.min(toDisp(q) - toDisp(lim[0]), toDisp(lim[1]) - toDisp(q)) : null,
      vlim: meta.velocity_limit?.[i], vmax: meta.max_velocity?.[i],
      eff, ext, model: (ext !== null && ext !== undefined) ? ext + eff : null, effLim, effFrac: Math.abs(eff) / (effLim || 1),
      effUnit: finger ? "N" : "Nm",
      kp: soft && !finger ? 0 : live.kp?.[i], kd: soft && !finger ? meta.soft_kd?.[i] : live.kd?.[i], soft,
      mos: h.temp_mos?.[k], rotor: h.temp_rotor?.[k], status: h.motor_status?.[k],
      motorId: motor ? motor[0] : null, motorType: motor ? motor[1] : null, channel: h.channel,
      owner: live.busy?.[group] ?? null, grip: finger ? live.grippers?.[arm] : null,
      passive: meta.passive,
    };
  }

  const n = (v, d = 1) => fmt(v, d);
  function stateCell(j) {
    const [label, cls, tipText] = motorState(j.status);
    return { label, cls, tipText };
  }

  function rows(j, full) {
    if (j.tcp) return tcpRows(j);
    const ms = stateCell(j);
    const r = [];
    const add = (k, v) => r.push(`<span>${k}</span><span>${v}</span>`);
    add("motor", `<span class="mstate ${ms.cls}">${ms.label}</span>`);
    if (j.motorType) add("model · CAN id", `${j.motorType} · 0x${j.motorId.toString(16).padStart(2, "0")} on ${j.channel}`);
    add("position", `${n(j.pos, 2)} ${j.unit}` + (j.finger ? "" : ` (${n(j.q, 4)} rad)`));
    add("commanded", `${n(j.ref, 2)} ${j.unit}`);
    if (full) add("sent to motor", `${n(j.cmd, 2)} ${j.unit} (after guards)`);
    add("tracking error", `${n(j.err, 2)} ${j.unit}` + (j.finger ? "" : ` · guard ±${n(GUARD_TRACK_DEG, 1)}°`));
    if (j.lo !== null) add("range", `${n(j.lo, 1)} … ${n(j.hi, 1)} ${j.unit} · ${n(100 * j.frac, 0)} %`);
    if (j.toLimit !== null) add("to nearest limit", `${n(j.toLimit, 1)} ${j.unit}`);
    add("velocity", `${n(j.vel, 1)} ${j.unit}/s` + (full && j.vlim ? ` · cap ${n(j.finger ? 2000 * j.vlim : deg(j.vlim), 0)}` : ""));
    add("effort (measured)", `${n(j.eff, 3)} ${j.effUnit} · ${n(100 * j.effFrac, 0)} % of ${j.effLim}`);
    if (!j.finger && !j.passive) {
      add("model feed-forward", `${n(j.model, 3)} Nm`);
      add("external (estimated)", `${n(j.ext, 3)} Nm`);
    }
    if (j.grip) add("gripper", `${n(j.grip.width * 1000, 1)} mm · ${n(j.grip.effort, 1)} N${j.grip.stalled ? " · stalled" : ""}`);
    if (!j.passive) add("gains kp · kd", `${n(j.kp, 1)} · ${n(j.kd, 2)}${j.soft ? " (soft)" : ""}`);
    add("temperature", `MOSFET ${n(j.mos, 0)} °C · rotor ${n(j.rotor, 0)} °C`);
    add("running command", j.owner !== null ? `#${j.owner}` : "none");
    return r.join("");
  }

  // ---------------------------------------------------------------- hover card
  function placeTip() {
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = mouse[0] + 16, y = mouse[1] + 16;
    if (x + w > window.innerWidth - 8) x = mouse[0] - w - 16;
    if (y + h > window.innerHeight - 8) y = window.innerHeight - h - 8;
    tip.style.left = x + "px"; tip.style.top = Math.max(8, y) + "px";
  }
  function paintTip() {
    if (!hovered || !live || !meta) return;
    const j = info(hovered.arm, hovered.k);
    tip.innerHTML = `<h5>${j.label} <span class="dim">· ${j.role}</span></h5><div class="kv">${rows(j, false)}</div>
      <div class="hint">click to open in a tab on the right</div>`;
    placeTip();
  }
  function hover(arm, k, e) { hovered = { arm, k }; mouse = [e.clientX, e.clientY]; tip.classList.remove("hidden"); paintTip(); }
  function move(e) { mouse = [e.clientX, e.clientY]; if (hovered) placeTip(); }
  function leave() { hovered = null; tip.classList.add("hidden"); }

  // ---------------------------------------------------------------- tabs
  const tabbar = $("tabbar"), side = $("side");
  tabbar.querySelector('[data-tab="log"]').addEventListener("click", () => select("log"));

  function key(arm, k) { return `${arm}:${k}`; }
  function select(id) {
    active = id;
    for (const b of tabbar.children) b.classList.toggle("active", b.dataset.tab === id);
    for (const p of side.querySelectorAll(".pane")) p.classList.toggle("active", p.dataset.tab === id || (id === "log" && p.id === "pane-log"));
    lastPaint = 0;
  }
  function open(arm, k) {
    const id = key(arm, k);
    if (!tabs.has(id)) {
      const btn = el("button", "tab");
      btn.dataset.tab = id;
      const short = k === "tcp" ? "tcp" : k < 6 ? "J" + (k + 1) : "grip";
      btn.innerHTML = `<span class="dot"></span>${arm === "left" ? "L" : "R"} ${short}<span class="x" title="close">×</span>`;
      btn.addEventListener("click", (e) => { if (e.target.classList.contains("x")) close(id); else select(id); });
      tabbar.append(btn);
      const pane = el("div", "pane jdetail");
      pane.dataset.tab = id;
      pane.innerHTML = k === "tcp" ? `<h3></h3><div class="sub"></div><div class="kv"></div>
        <h4>force on the gripper <span class="legend">(<b style="color:#ff6b6b">x</b>, <b style="color:#6bd36b">y</b>, <b style="color:#5aa9ff">z</b>, <b style="color:#d8dee6">|F|</b>; world frame; last ${HISTORY_S} s)</span></h4><canvas class="chart" data-c="force"></canvas>
        <h4>torque about the tip <span class="legend">(<b style="color:#ff6b6b">x</b>, <b style="color:#6bd36b">y</b>, <b style="color:#5aa9ff">z</b>)</span></h4><canvas class="chart" data-c="torque"></canvas>`
        : `<h3></h3><div class="sub"></div><div class="kv"></div>
        <h4>position · commanded <span class="legend">(<b style="color:#d8dee6">measured</b>, <b style="color:#4cc3ff">commanded</b>, last ${HISTORY_S} s)</span></h4><canvas class="chart" data-c="pos"></canvas>
        <h4>velocity</h4><canvas class="chart" data-c="vel"></canvas>
        <h4>effort <span class="legend">(dashed: ${GUARD_EFFORT * 100} % of the motor limit, the guard)</span></h4><canvas class="chart" data-c="eff"></canvas>
        <h4>temperature <span class="legend">(<b style="color:#f2b33d">MOSFET</b>, <b style="color:#ff8a65">rotor</b>; dashed 50 / 60 °C)</span></h4><canvas class="chart" data-c="temp"></canvas>`;
      side.append(pane);
      tabs.set(id, { arm, k, btn, pane });
    }
    select(id);
  }
  function close(id) {
    const t = tabs.get(id);
    if (!t) return;
    t.btn.remove(); t.pane.remove(); tabs.delete(id);
    if (active === id) select("log");
  }
  function isOpen(arm, k) { return tabs.has(key(arm, k)); }

  // ---------------------------------------------------------------- charts
  function chart(canvas, series, opts = {}) {
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
    if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const vals = series.flatMap((s) => s.ys.filter((v) => v !== null && Number.isFinite(v)));
    const lines = (opts.lines || []).filter((v) => v !== null && Number.isFinite(v));
    if (!vals.length) { g.fillStyle = "#5f6b78"; g.font = "11px sans-serif"; g.fillText("no data", 8, 16); return; }
    let lo = Math.min(...vals, ...(opts.includeLines ? lines : [])), hi = Math.max(...vals, ...(opts.includeLines ? lines : []));
    if (opts.minSpan && hi - lo < opts.minSpan) { const c = (hi + lo) / 2; lo = c - opts.minSpan / 2; hi = c + opts.minSpan / 2; }
    const pad = (hi - lo) * 0.12 || 1; lo -= pad; hi += pad;
    const t1 = hist.length ? hist[hist.length - 1].t : 0, t0 = t1 - HISTORY_S;
    const X = (t) => ((t - t0) / HISTORY_S) * (W - 46) + 2, Y = (v) => H - 4 - ((v - lo) / (hi - lo)) * (H - 8);
    g.strokeStyle = "#5f6b78"; g.setLineDash([3, 3]); g.lineWidth = 1;
    for (const v of lines) { if (v > lo && v < hi) { g.beginPath(); g.moveTo(0, Y(v)); g.lineTo(W - 44, Y(v)); g.stroke(); } }
    g.setLineDash([]);
    for (const s of series) {
      g.strokeStyle = s.color; g.lineWidth = 1.5; g.beginPath();
      let started = false;
      s.ys.forEach((v, idx) => {
        if (v === null || !Number.isFinite(v)) { started = false; return; }
        const x = X(hist[idx].t), y = Y(v);
        if (started) g.lineTo(x, y); else { g.moveTo(x, y); started = true; }
      });
      g.stroke();
    }
    g.fillStyle = "#8a96a3"; g.font = "10px ui-monospace, monospace"; g.textAlign = "right";
    g.fillText(fmt(hi - pad, opts.digits ?? 1), W - 3, 11);
    g.fillText(fmt(lo + pad, opts.digits ?? 1), W - 3, H - 4);
    const last = series[0].ys[series[0].ys.length - 1];
    g.fillStyle = series[0].color; g.fillText(fmt(last, opts.digits ?? 1), W - 3, H / 2 + 3);
  }

  function paintTcpTab(t) {
    const j = tcpInfo(t.arm);
    t.btn.querySelector(".dot").style.background = j.contact ? "#ff8a3d" : "#6c7a89";
    t.pane.querySelector("h3").innerHTML = `${j.label} <span class="dim">· ${j.role}</span>`;
    t.pane.querySelector(".sub").textContent = "Push or pull the gripper: the arrow in the 3D view and these show the force on it."
      + " Zero it (\"zero force\" in the 3D view) where you start; the zero drifts as the arm moves.";
    t.pane.querySelector(".kv").innerHTML = tcpRows(j);
    const w = (h, idx) => (h.ee?.[t.arm] ? -h.ee[t.arm].wrench[idx] : null);
    const c = (name) => t.pane.querySelector(`[data-c="${name}"]`);
    const cols = ["#ff6b6b", "#6bd36b", "#5aa9ff"];
    chart(c("force"), [0, 1, 2].map((idx) => ({ ys: hist.map((h) => w(h, idx)), color: cols[idx] }))
      .concat([{ ys: hist.map((h) => h.ee?.[t.arm] ? Math.hypot(w(h, 0), w(h, 1), w(h, 2)) : null), color: "#d8dee6" }]),
      { lines: [0], minSpan: 4, digits: 1 });
    chart(c("torque"), [3, 4, 5].map((idx) => ({ ys: hist.map((h) => w(h, idx)), color: cols[idx - 3] })),
      { lines: [0], minSpan: 0.4, digits: 2 });
  }

  function paintTab(t) {
    if (t.k === "tcp") return paintTcpTab(t);
    const j = info(t.arm, t.k);
    const ms = stateCell(j);
    t.btn.querySelector(".dot").style.background = MOTOR_RING[ms.cls] === "#0f1216" ? "#6c7a89" : MOTOR_RING[ms.cls];
    t.pane.querySelector("h3").innerHTML = `${j.label} <span class="dim">· ${j.role}</span>`;
    t.pane.querySelector(".sub").textContent = `${j.name}${j.passive ? " · monitor mode (motors off)" : ""} · ${ms.tipText}`;
    t.pane.querySelector(".kv").innerHTML = rows(j, true);
    const i = j.i, conv = j.finger ? (v) => 2000 * v : deg;
    const pick = (f) => hist.map(f);
    const c = (name) => t.pane.querySelector(`[data-c="${name}"]`);
    chart(c("pos"), [{ ys: pick((h) => conv(h.q[i])), color: "#d8dee6" }, { ys: pick((h) => conv(h.qr[i])), color: "#4cc3ff" }],
      { lines: [j.lo, j.hi], minSpan: j.finger ? 4 : 2, digits: 1 });
    chart(c("vel"), [{ ys: pick((h) => conv(h.dq[i])), color: "#9fd18b" }], { minSpan: j.finger ? 10 : 5, digits: 0 });
    chart(c("eff"), [{ ys: pick((h) => h.eff[i]), color: "#c49bff" }],
      { lines: [GUARD_EFFORT * j.effLim, -GUARD_EFFORT * j.effLim], minSpan: j.effLim * 0.1, digits: 2 });
    const tk = (h, kind) => h.temps?.[t.arm]?.[kind]?.[t.k] ?? null;
    chart(c("temp"), [{ ys: pick((h) => tk(h, "temp_mos")), color: "#f2b33d" }, { ys: pick((h) => tk(h, "temp_rotor")), color: "#ff8a65" }],
      { lines: [50, 60], minSpan: 4, digits: 0 });
  }

  // ---------------------------------------------------------------- live
  function onLive(l, m) {
    live = l; meta = m;
    const now = Date.now() / 1000;
    const temps = {};
    for (const [a, x] of Object.entries(l.health?.backend?.arms || {})) temps[a] = { temp_mos: x.temp_mos, temp_rotor: x.temp_rotor };
    hist.push({ t: now, q: l.q, qr: l.q_ref, dq: l.dq, eff: l.effort, temps, ee: l.ee });
    while (hist.length && hist[0].t < now - HISTORY_S) hist.shift();
    if (performance.now() - lastPaint < 100) return;               // 10 Hz is plenty for text and charts
    lastPaint = performance.now();
    paintTip();
    for (const [id, t] of tabs) {
      if (id === active) paintTab(t);
      else if (t.k !== "tcp") { const ms = stateCell(info(t.arm, t.k)); t.btn.querySelector(".dot").style.background = MOTOR_RING[ms.cls] === "#0f1216" ? "#6c7a89" : MOTOR_RING[ms.cls]; }
    }
  }

  document.addEventListener("keydown", (e) => { if (e.key === "Escape") leave(); });
  return { hover, move, leave, open, close, isOpen, onLive, select };
})();
