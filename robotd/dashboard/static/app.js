// robotd dashboard: polling, the stop bar, cameras, the joint diagram, info and the log.
// A classic script on purpose: the stop buttons must work even when the 3D module (three.js from a CDN) fails.
"use strict";

const $ = (id) => document.getElementById(id);
const ROLE = ["base yaw", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3"];
const TEMP_WARN = 45, TEMP_HOT = 50;          // deg C (a DM4310 latched over-temperature at a 53 C rotor on the cell)
const LIVE_PERIOD_MS = 50;

let meta = null, metaSeq = -1, live = null, dashOk = true, liveFails = 0, lastPanels = 0;
let camSeq = -1, logSeq = 0, lastAction = null, camNames = "", viewSeq = -1;

function timeout(ms) { const c = new AbortController(); setTimeout(() => c.abort(), ms); return c.signal; }
const deg = (r) => (r * 180 / Math.PI);
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function tempClass(t) { return t === null || t === undefined ? "" : t >= TEMP_HOT ? "hot" : t >= TEMP_WARN ? "warn" : "ok"; }

// ------------------------------------------------------------------ the operator's requests
function setMsg(text, cls) { const m = $("estop-msg"); m.textContent = text; m.className = "estop-msg " + (cls || ""); }

async function act(name, retry = true) {
  const NAME = name.toUpperCase();
  const t0 = performance.now();
  lastAction = { name, t: Date.now() };
  setMsg(`${NAME} sent…`);
  try {
    const r = await fetch("api/action/" + name, { method: "POST", headers: { "X-Dashboard": "1" }, signal: timeout(2500) });
    const j = await r.json();
    const rt = performance.now() - t0;
    if (j.ok) setMsg(`${NAME}: done — the driver acknowledged in ${fmt(j.ms, 1)} ms (round trip ${fmt(rt, 0)} ms)`, "ok");
    else setMsg(`${NAME} refused: ${j.error}`, "err");
    return j;
  } catch (e) {
    if (retry && (name === "freeze" || name === "soft")) return act(name, false);
    setMsg(`${NAME} could not reach the dashboard server. If the arm is unsafe: power switch (the arms drop).`, "err");
  }
}

function arm(button, name) {                   // two clicks within 3 s for actions that move or re-enable the arms
  let timer = null;
  const label = button.querySelector("small"), plain = label.textContent;
  const disarm = () => { button.classList.remove("armed"); label.textContent = plain; clearTimeout(timer); timer = null; };
  button.addEventListener("click", () => {
    if (button.classList.contains("armed")) { disarm(); act(name); return; }
    button.classList.add("armed"); label.textContent = "click again to confirm";
    timer = setTimeout(disarm, 3000);
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") disarm(); });
}

$("btn-freeze").addEventListener("pointerdown", (e) => { e.preventDefault(); act("freeze"); });
$("btn-soft").addEventListener("pointerdown", (e) => { e.preventDefault(); act("soft"); });
arm($("btn-park"), "park");
$("btn-zero").addEventListener("click", () => act("zero"));
arm($("btn-release"), "release");
arm($("btn-off"), "off");
document.addEventListener("keydown", (e) => {
  const t = e.target;                                   // typing in the editor or a text box is not a shortcut
  if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && !["checkbox", "range", "button"].includes(t.type)))) return;
  if (e.code === "Space") { e.preventDefault(); if (!e.repeat) act("freeze"); }
  else if ((e.key === "s" || e.key === "S") && !e.repeat && !e.metaKey && !e.ctrlKey) { e.preventDefault(); act("soft"); }
}, true);

// ------------------------------------------------------------------ polling
async function pollLive() {
  const t0 = performance.now();
  try {
    const r = await fetch("api/live", { cache: "no-store", signal: timeout(1500) });
    live = await r.json();
    dashOk = true; liveFails = 0;
    onLive(performance.now() - t0);
  } catch (e) {
    if (++liveFails >= 3) { dashOk = false; renderState(); }
  }
  setTimeout(pollLive, Math.max(0, LIVE_PERIOD_MS - (performance.now() - t0)));
}

async function loadMeta() {
  try {
    const j = await (await fetch("api/meta", { cache: "no-store" })).json();
    meta = j.meta; metaSeq = j.meta_seq;
    if (meta) { camNames = ""; buildArms(); }
    $("top-robot").textContent = meta ? (meta.robot || "") : "";
  } catch (e) { /* retried on the next live update */ }
}

async function pollLog() {
  try {
    const j = await (await fetch("api/events?since=" + logSeq, { cache: "no-store", signal: timeout(2000) })).json();
    for (const ev of j.events) addLog(ev);
    logSeq = j.seq;
  } catch (e) { /* the live poll reports the outage */ }
  setTimeout(pollLog, 250);
}

function onLive(rtt) {
  window.__live = live; window.__meta = meta;
  if (window.Joints && live.connected && meta) Joints.onLive(live, meta);
  window.dispatchEvent(new CustomEvent("robot-live", { detail: live }));
  if (live.meta_seq !== undefined && live.meta_seq !== metaSeq && live.meta_seq > 0) loadMeta();
  renderTop(rtt);
  renderState();
  if (live.connected && meta) {
    const views = (live.views || []).slice(-2).map(v => "view:" + v.name);     // the newest images programs showed
    const names = [...new Set([...meta.cameras, ...Object.keys(live.cams || {})]), ...views];
    if (names.join(",") !== camNames) { camNames = names.join(","); buildCams(names); camSeq = -1; viewSeq = -1; }
  }
  if (live.connected && live.cam_seq !== camSeq) { camSeq = live.cam_seq; refreshCams(); }
  if (live.connected) { refreshViews(live.view_seq !== viewSeq); renderViewsPane(); }
  viewSeq = live.view_seq;
  const now = performance.now();
  if (now - lastPanels > 100) { lastPanels = now; if (live.connected && meta) { renderArms(); renderInfo(); } }
}

// ------------------------------------------------------------------ top bar, banner, badge
function renderTop(rtt) {
  const b = $("top-backend");
  if (live.connected && meta) {
    b.textContent = meta.backend; b.className = "chip ok";
    $("top-mode").textContent = meta.lockstep ? "lockstep sim" : (live.health?.backend?.sim ? "realtime sim" : "realtime");
  } else { b.textContent = "no driver"; b.className = "chip bad"; $("top-mode").textContent = "—"; }
  $("top-t").textContent = live.connected ? fmt(live.t, 2) + " s" : "—";
  $("top-hz").textContent = live.tick_hz ? fmt(live.tick_hz, 0) + " Hz" : "—";
  const L = live.connected ? live.health?.loop : null, tick = $("top-tick");
  tick.textContent = L ? `${fmt(L.period_ms.p99, 1)} ms` + (L.stalls_window ? ` · ${L.stalls_window} stalls` : "") : "—";
  tick.style.color = L && L.stalls_window ? "var(--warn)" : "";
  $("top-poll").textContent = fmt(rtt, 0) + " ms";
  $("top-cams").textContent = live.cam_age !== null && live.cam_age !== undefined ? fmt(live.cam_age, 1) + " s ago" : "—";
}

function stateOf() {
  if (!dashOk) return ["down", "DASHBOARD OFFLINE", "The dashboard server does not answer. The stop buttons cannot reach the robot from this page."];
  if (!live || !live.connected) return ["down", "NO DRIVER", "robotd does not answer. If it died, the motors time out within ~0.4 s and the arms go limp."];
  const f = live.fault;
  if (meta && meta.passive && !f) return ["monitor", "MONITOR · motors off", null];
  if (f && f.operator && f.soft) return ["soft", "SOFT · operator", "SOFT: the arms float on gravity compensation and yield to a push, the grippers are limp. Move them clear, then RELEASE (they hold stiff)."];
  if (f && f.operator) return ["freeze", "FROZEN · operator", "FROZEN: every motion stopped, the arms hold stiff. Programs are refused until you RELEASE."];
  if (live.parked) return ["parked", "PARKED · operator", "PARKED: the arms go to (or hold at) the park pose. Programs are refused until you RELEASE."];
  if (f) return ["fault", "FAULT", "FAULT: " + f.reason + " — the arms hold; a program's recover() or RELEASE clears it."];
  const busy = Object.values(live.busy || {}).some((v) => v !== null);
  return busy ? ["running", "MOVING", null] : ["idle", "IDLE · holding", null];
}

function healthAlarm() {
  for (const [a, g] of Object.entries(live?.grippers || {})) {
    const i = meta?.joints.indexOf(meta.finger_joint[a]);
    const lim = i >= 0 ? meta.effort_limit[i] : 50;
    if (g.relaxed) return `${a} gripper was relaxed: it pushed far over its limit (jammed or blocked). It stays limp until its next command.`;
    if (Math.abs(g.effort) > 1.2 * lim) return `${a} gripper is pushing ${fmt(Math.abs(g.effort), 0)} N (limit ${lim} N): jammed or blocked? FREEZE / SOFT, or free it by hand.`;
  }
  const arms = live?.health?.backend?.arms;
  if (!arms) return null;
  for (const [a, h] of Object.entries(arms)) {
    if (h.running === false) return `${a} arm: i2rt's control loop stopped — that arm is not controlled (motor error or overheating).`;
    const bad = (h.motor_status || []).map((s, k) => [k, s]).filter(([, s]) => s && s.startsWith("error"));
    if (bad.length) return `${a} arm motor fault: ` + bad.map(([k, s]) => `${k < 6 ? "J" + (k + 1) : "gripper"} ${s.slice(7)}`).join(", ");
    const temps = [...(h.temp_mos || []), ...(h.temp_rotor || [])];
    const hot = Math.max(...temps.filter((t) => t !== null), -1);
    if (hot >= TEMP_HOT) return `${a} arm motor at ${fmt(hot, 0)} °C: park and let it cool.`;
  }
  return null;
}

function renderState() {
  const [cls, text, banner] = stateOf();
  const badge = $("state-badge");
  badge.textContent = text; badge.className = "badge " + cls;
  $("estop").className = ["freeze", "soft", "down"].includes(cls) ? cls : "";
  const bn = $("view-banner"), alarm = live && live.connected ? healthAlarm() : null;
  const show = banner || alarm;
  bn.classList.toggle("hidden", !show);
  if (show) { bn.textContent = banner || alarm; bn.className = "banner " + (banner ? cls : "fault"); }
}

// ------------------------------------------------------------------ cameras
function buildCams(names) {
  const box = $("cams"); box.innerHTML = "";
  for (const name of names) {
    const isView = name.startsWith("view:");
    const tile = el("div", isView ? "cam view" : "cam"); tile.dataset.name = name;
    const img = el("img"); img.alt = name;
    const empty = el("div", "cam-empty", isView ? "loading…" : "waiting for frames…");
    img.onload = () => { empty.classList.add("hidden"); img.classList.remove("hidden"); };
    img.onerror = () => { img.classList.add("hidden"); empty.classList.remove("hidden"); };
    tile.append(img, empty, el("span", "cam-label", isView ? "▸ " + name.slice(5) : name), el("span", "cam-age", ""));
    tile.title = isView ? "an image a program showed (robot.viz.show)" : (meta.camera_info?.[name] || name);
    tile.addEventListener("click", () => tile.classList.toggle("big"));
    box.append(tile);
  }
}

function refreshCams() {
  for (const tile of $("cams").children) {
    if (tile.dataset.name.startsWith("view:")) continue;
    const empty = tile.querySelector(".cam-empty");
    if (live.cam_error) { empty.textContent = live.cam_error; empty.classList.remove("hidden"); continue; }
    tile.querySelector("img").src = `api/camera/${encodeURIComponent(tile.dataset.name)}.jpg?s=${camSeq}`;
    const ci = live.cams?.[tile.dataset.name];
    const age = tile.querySelector(".cam-age");
    age.textContent = ci && !ci.pose ? "no pose · no cloud" : "#" + camSeq;
    age.title = ci && !ci.pose ? (ci.pose_error || "") : "";
  }
}

// ------------------------------------------------------------------ the Views tab and the side panel
let paneSeq = -1;
function viewAge(v) {
  const age = Math.max(0, (live.wall_now || Date.now() / 1000) - v.wall);
  return age < 90 ? fmt(age, 0) + " s ago" : age < 5400 ? fmt(age / 60, 0) + " min ago" : fmt(age / 3600, 1) + " h ago";
}
function renderViewsPane() {
  const list = $("views-list"), views = [...(live.views || [])].reverse();
  $("views-count").textContent = views.length ? `${views.length} shown` : "";
  if (live.view_seq !== paneSeq) {
    const fresh = paneSeq >= 0 && live.view_seq > paneSeq;
    paneSeq = live.view_seq;
    if (fresh && !$("pane-views").classList.contains("active")) $("views-dot").style.background = "var(--accent)";
    if (!views.length) return;
    const old = new Map([...list.querySelectorAll(".view-item")].map(x => [x.dataset.name, x]));
    list.textContent = "";
    for (const v of views) {
      let item = old.get(v.name);
      if (!item) {
        item = el("div", "view-item"); item.dataset.name = v.name;
        const head = el("div", "vh"); head.append(el("b", null, v.name), el("span", "age", ""), el("span", "note", ""));
        const img = el("img"); img.alt = v.name;
        img.addEventListener("click", () => window.open(img.src, "_blank"));
        item.append(head, img);
      }
      const img = item.querySelector("img");
      if (img.dataset.seq !== String(v.seq)) { img.dataset.seq = String(v.seq); img.src = `api/view/${encodeURIComponent(v.name)}.jpg?s=${v.seq}`; }
      item.querySelector(".note").textContent = (v.note || "") + (v.client ? `  · ${v.client}` : "");
      item.classList.toggle("new", v.seq === live.view_seq);
      list.append(item);
    }
  }
  for (const item of list.querySelectorAll(".view-item")) {
    const v = (live.views || []).find(x => x.name === item.dataset.name);
    if (v) item.querySelector(".age").textContent = viewAge(v);
  }
}
document.querySelector('#tabbar [data-tab="views"]').addEventListener("click", () => {
  if (window.Joints) Joints.select("views");
  $("views-dot").style.background = "";
});
function setSide(on) {
  $("app").classList.toggle("noside", !on);
  try { localStorage.setItem("robotd-dash-side", on ? "1" : "0"); } catch (e) { /* private window: not remembered */ }
}
$("side-toggle").addEventListener("click", () => setSide($("app").classList.contains("noside")));
document.addEventListener("keydown", (e) => {
  if (e.key !== "\\" || e.target.closest("input, textarea, select, [contenteditable]")) return;
  setSide($("app").classList.contains("noside"));
});
try { if (localStorage.getItem("robotd-dash-side") === "0") setSide(false); } catch (e) { /* not remembered */ }

function refreshViews(reload) {
  const byName = Object.fromEntries((live.views || []).map(v => [v.name, v]));
  for (const tile of $("cams").children) {
    if (!tile.dataset.name.startsWith("view:")) continue;
    const v = byName[tile.dataset.name.slice(5)];
    if (!v) continue;
    const img = tile.querySelector("img");
    if (reload || img.dataset.seq !== String(v.seq)) {
      img.dataset.seq = String(v.seq);
      img.src = `api/view/${encodeURIComponent(v.name)}.jpg?s=${v.seq}`;
    }
    const age = Math.max(0, (live.wall_now || Date.now() / 1000) - v.wall);
    tile.querySelector(".cam-age").textContent = (age < 90 ? fmt(age, 0) + " s" : fmt(age / 60, 0) + " min") + " ago";
    tile.title = (v.note ? v.note + " · " : "") + (v.client ? "from " + v.client : "");
  }
}

// ------------------------------------------------------------------ joint diagram
const armViews = {};

function buildArms() {
  for (const arm of meta.arms) {
    const card = $("arm-" + arm);
    if (!card) continue;
    card.innerHTML = "";
    const side = el("div", "arm-side");
    side.append(el("div", "arm-title", arm + " arm"));
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "-0.18 -0.66 0.86 0.74");
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.innerHTML = `<line x1="-0.18" y1="0" x2="0.68" y2="0" stroke="#3a4552" stroke-width="0.006"/>
      <polyline class="links" fill="none" stroke="#9fb0c2" stroke-width="0.022" stroke-linejoin="round" stroke-linecap="round"/>
      <rect class="base" x="-0.05" y="-0.02" width="0.1" height="0.04" rx="0.008" fill="#2f3a46"/>
      <g class="joints"></g><circle class="tcp" r="0.012" fill="#4cc3ff"/>`;
    const g = svg.querySelector(".joints");
    for (let k = 0; k < 6; k++) {
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("r", "0.019"); c.setAttribute("stroke", "#0f1216"); c.setAttribute("stroke-width", "0.006");
      hookJoint(c, arm, k);
      g.append(c);
    }
    side.append(svg);
    const table = el("table");
    table.innerHTML = `<thead><tr><th>joint</th><th>deg</th><th class="col-w" title="commanded minus measured">Δ°</th>
      <th class="barcell">range · ref</th><th class="col-w">°/s</th><th class="col-eff">effort</th>
      <th class="tc">°C mos/rot</th><th>motor</th></tr></thead>`;
    const body = el("tbody");
    const names = [...meta.arm_joints[arm], meta.finger_joint[arm]];
    const rows = names.map((n, k) => {
      const tr = el("tr");
      const label = k < 6 ? `J${k + 1}` : "grip";
      tr.innerHTML = `<td class="name">${label}<small>${k < 6 ? ROLE[k] : "mm"}</small></td><td class="v"></td>
        <td class="err col-w"></td>
        <td class="barcell"><span class="bar"><span class="ref"></span><span class="mark"></span></span></td>
        <td class="w col-w"></td>
        <td class="col-eff"><span class="bar eff"><span class="fill"></span></span><span class="effv"></span></td>
        <td class="tc"><span class="temp"></span></td><td class="ms"><span class="mstate">—</span></td>`;
      hookJoint(tr, arm, k);
      body.append(tr);
      return { tr, i: meta.joints.indexOf(n), k };
    });
    const tcp = el("tr", "tcprow");                // the force on the gripper, from all six joints
    tcp.innerHTML = `<td class="name">tcp<small>force</small></td><td class="fmag" colspan="2"></td><td class="fvec" colspan="5"></td>`;
    hookJoint(tcp, arm, "tcp");
    body.append(tcp);
    table.append(body);
    card.append(side, table);
    armViews[arm] = { svg, rows, tcp };
  }
}

// Each DM motor reports a status in every reply: disabled, enabled, or a fault (DM's codes 0x8-0xE).
const MOTOR_RING = { on: "#3ccf7a", off: "#0f1216", odd: "#4c6a86", fault: "#ff5a4f", none: "#0f1216" };
function motorState(s) {
  if (!s) return ["—", "none", "no motor status (simulation)"];
  if (s === "enabled") return ["on", "on", "enabled: the motor applies torque"];
  if (s === "disabled") return ["off", "off", "disabled: no torque, the joint is free"];
  if (s.startsWith("error")) return ["FAULT", "fault", s + " — the motor has shut itself off"];
  return [s.replace("status ", ""), "odd", s + ": a status DM does not document; not a fault (disabled DM4310s report it)"];
}

function hookJoint(node, arm, k) {          // hover: the full card (joints.js); click: a tab on the right
  node.addEventListener("mouseenter", (e) => window.Joints && Joints.hover(arm, k, e));
  node.addEventListener("mousemove", (e) => window.Joints && Joints.move(e));
  node.addEventListener("mouseleave", () => window.Joints && Joints.leave());
  node.addEventListener("click", () => window.Joints && Joints.open(arm, k));
}

function placeOnBar(bar, sel, v, lo, hi) {
  const x = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  bar.querySelector(sel).style.left = `calc(${(x * 100).toFixed(1)}% - 1px)`;
}

function renderArms() {
  const temps = live.health?.backend?.arms || {};
  for (const [arm, view] of Object.entries(armViews)) {
    const pts = live.diagram?.[arm];
    const t = temps[arm] || {};
    if (pts) {
      const xy = pts.map(([h, z]) => [h, -z]);
      view.svg.querySelector(".links").setAttribute("points", xy.map((p) => p.join(",")).join(" "));
      const circles = view.svg.querySelectorAll(".joints circle");
      circles.forEach((c, k) => {
        const p = xy[k + 1];
        c.setAttribute("cx", p[0]); c.setAttribute("cy", p[1]);
        const tc = tempClass(t.temp_rotor?.[k] ?? t.temp_mos?.[k]);
        c.setAttribute("fill", tc === "hot" ? "#ff5a4f" : tc === "warn" ? "#f2b33d" : tc === "ok" ? "#3ccf7a" : "#6c7a89");
        c.setAttribute("stroke", MOTOR_RING[motorState(t.motor_status?.[k])[1]]);     // the motor's state, like its LED
      });
      const tcp = view.svg.querySelector(".tcp"), end = xy[xy.length - 1];
      tcp.setAttribute("cx", end[0]); tcp.setAttribute("cy", end[1]);
    }
    const e = live.ee?.[arm];
    if (e) {                                         // the API's wrench is what the tool applies; on it = minus that
      const F = e.wrench.slice(0, 3).map((v) => -v), mag = Math.hypot(...F);
      view.tcp.querySelector(".fmag").textContent = meta.passive ? "— (motors off)" : `${fmt(mag, 1)} N`;
      view.tcp.querySelector(".fvec").textContent = meta.passive ? "" :
        `x ${fmt(F[0], 1)}  y ${fmt(F[1], 1)}  z ${fmt(F[2], 1)}${e.contact ? "  contact" : ""}`;
      view.tcp.classList.toggle("open", !!(window.Joints && Joints.isOpen(arm, "tcp")));
    }
    for (const { tr, i, k } of view.rows) {
      if (i < 0) continue;
      const q = live.q[i], qr = live.q_ref[i], lim = meta.limits?.[i];
      const isFinger = k === 6;
      tr.querySelector(".v").textContent = isFinger ? fmt((live.grippers?.[arm]?.width ?? 2 * q) * 1000, 1) : fmt(deg(q), 1);
      const bar = tr.querySelector(".bar");
      if (lim) { placeOnBar(bar, ".mark", q, lim[0], lim[1]); placeOnBar(bar, ".ref", qr, lim[0], lim[1]); }
      tr.querySelector(".w").textContent = isFinger ? fmt(live.dq[i] * 2000, 0) : fmt(deg(live.dq[i]), 0);
      const err = isFinger ? 2000 * (qr - q) : deg(qr - q), errCell = tr.querySelector(".err");
      errCell.textContent = fmt(err, 1);
      errCell.className = "err col-w" + (isFinger ? "" : Math.abs(err) > 6 ? " hot" : Math.abs(err) > 2 ? " warn" : "");
      tr.querySelector(".effv").textContent = fmt(live.effort[i], isFinger ? 1 : 2);
      tr.classList.toggle("open", !!(window.Joints && Joints.isOpen(arm, k)));
      const frac = Math.min(1, Math.abs(live.effort[i]) / (meta.effort_limit[i] || 1));
      const fill = tr.querySelector(".fill");
      fill.style.width = (frac * 100).toFixed(0) + "%";
      fill.style.background = frac > 0.8 ? "var(--hot)" : frac > 0.6 ? "var(--warn)" : "var(--accent)";
      fill.parentElement.title = `${fmt(live.effort[i], 2)} ${isFinger ? "N" : "Nm"} of ${meta.effort_limit[i]}`;
      const tm = t.temp_mos?.[k], tr_ = t.temp_rotor?.[k];
      const chip = tr.querySelector(".temp");
      chip.textContent = tm === undefined && tr_ === undefined ? "—" : `${fmt(tm, 0)}/${fmt(tr_, 0)}`;
      chip.className = "temp " + tempClass(Math.max(tm ?? -1, tr_ ?? -1) >= 0 ? Math.max(tm ?? -1, tr_ ?? -1) : null);
      const [label, cls, tip] = motorState(t.motor_status?.[k]);
      const ms = tr.querySelector(".mstate");
      ms.textContent = label; ms.className = "mstate " + cls; ms.title = tip;
    }
  }
}

// ------------------------------------------------------------------ info
function row(label, value) { return `<div class="row"><span class="dim">${label}</span><b>${value}</b></div>`; }
// text a program can choose (its client name, a fault's reason) goes into the page escaped, never as markup
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function renderInfo() {
  const h = live.health || {}, hb = h.backend || {};
  let html = "<h4>driver</h4>";
  html += row("backend", meta.backend) + row("clock", meta.lockstep ? "lockstep" : "realtime");
  if (meta.passive) html += row("motors", "OFF (monitor)");
  html += row("control loop", live.tick_hz ? fmt(live.tick_hz, 0) + " Hz" : "—");
  const L = h.loop;                    // robotd's tick timing over the last 5 s (LoopStats in robotd/server.py)
  if (L) {
    const warn = (s, bad) => bad ? `<span style="color:var(--warn)">${s}</span>` : s;
    html += row("tick work", `${fmt(L.work_ms.mean, 2)} ms avg · ${fmt(L.busy_pct, 0)}% of ${fmt(L.target_ms, 0)} ms`);
    html += row("tick period p50/p99/max",
      warn(`${fmt(L.period_ms.p50, 1)} / ${fmt(L.period_ms.p99, 1)} / ${fmt(L.period_ms.max, 1)} ms`, L.period_ms.max > 20));
    html += row("lock wait p99/max", `${fmt(L.lock_ms.p99, 2)} / ${fmt(L.lock_ms.max, 1)} ms`);
    html += row("late >6 / stalls >20 ms", warn(`${L.late_window} / ${L.stalls_window} in ${fmt(L.window_s, 0)} s`,
      L.stalls_window > 0) + ` · ${L.late} / ${L.stalls} total`);
    // where a tick goes (Controller.sec): mean and p99 per phase, the slow ones first
    const ph = Object.entries(L.phases_ms || {}).sort((a, b) => b[1].mean - a[1].mean);
    for (const [name, v] of ph)
      if (v.mean >= 0.01 || v.p99 >= 0.1)
        html += row(`· ${name}`, warn(`${fmt(v.mean, 2)} ms avg · p99 ${fmt(v.p99, 2)}`, v.mean > 0.25 * L.target_ms));
  }
  html += row("state", live.fault ? esc(live.fault.reason) : "ok");
  const busy = Object.entries(live.busy || {}).filter(([, v]) => v !== null).map(([g, v]) => `${g} #${v}`);
  html += row("running", busy.length ? esc(busy.join(", ")) : "nothing");
  html += row("clients", esc((h.clients || []).filter((c) => !c.startsWith("dashboard")).join(", ") || "none"));
  html += "<h4>end effectors</h4>";
  for (const a of meta.arms) {
    const e = live.ee?.[a]; if (!e) continue;
    const p = [e.T[0][3], e.T[1][3], e.T[2][3]].map((v) => fmt(v * 1000, 0)).join(", ");
    const f = Math.hypot(...e.wrench.slice(0, 3));
    const g = live.grippers?.[a] || {};
    const force = meta.passive ? "— (motors off)" : `${fmt(f, 1)} N${e.contact ? " contact" : ""}`;   // no torque, no estimate
    html += row(`${a} tcp mm`, p) + row(`${a} force · grip`, `${force} · ${fmt(g.width * 1000, 0)} mm`);
  }
  if (hb.arms || hb.buses) {
    html += "<h4>hardware</h4>";
    for (const [a, x] of Object.entries(hb.arms || {})) {
      if (x.running !== null && x.running !== undefined) html += row(`${a} motor loop`, x.running ? "running" : "STOPPED");
      if (x.comm_hz) html += row(`${a} CAN loop (i2rt)`, `${fmt(x.comm_hz, 0)} Hz`);
      if (x.motor_status) {
        const bad = x.motor_status.map((s, k) => [k, s]).filter(([, s]) => s && s.startsWith("error"));
        const kinds = [...new Set(x.motor_status.filter((s) => s && !s.startsWith("error")))].join(" / ");
        html += row(`${a} motors`, bad.length ? bad.map(([k, s]) => `${k < 6 ? "J" + (k + 1) : "grip"} ${s}`).join(", ") : kinds || "—");
      }
      if (x.error) html += row(`${a} read error`, esc(x.error.slice(0, 40)));
    }
    for (const [ch, b] of Object.entries(hb.buses || {})) {
      html += row(ch, `${b.state || (b.up ? "up" : "down")} · rx ${fmt(b.rx_per_s, 0)}/s · err ${b.rx_errors ?? "—"}/${b.tx_errors ?? "—"}`);
    }
  }
  if (hb.sim) html += "<h4>simulation</h4>" + row("sim time", fmt(hb.sim_time, 2) + " s");
  $("info").innerHTML = html;
}

// ------------------------------------------------------------------ log
const logList = $("log-list");
let follow = true;
logList.addEventListener("scroll", () => {
  follow = logList.scrollTop + logList.clientHeight >= logList.scrollHeight - 8;
  if (follow) $("log-follow").classList.add("hidden");
});
$("log-follow").addEventListener("click", () => { logList.scrollTop = logList.scrollHeight; });
$("log-clear").addEventListener("click", () => { logList.innerHTML = ""; });
$("log-commands").addEventListener("change", (e) => logList.classList.toggle("no-commands", !e.target.checked));
$("log-clients").addEventListener("change", (e) => logList.classList.toggle("no-clients", !e.target.checked));
const styleNoCmd = document.createElement("style");
styleNoCmd.textContent = "#log-list.no-commands .ev.command, #log-list.no-commands .ev.finish:not(.aborted) { display: none; }"
  + " #log-list.no-clients .ev.client { display: none; }";
document.head.append(styleNoCmd);

function addLog(ev) {
  const d = new Date(ev.wall * 1000);
  const ts = d.toLocaleTimeString([], { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
  const key = ev.kind + "|" + ev.text;
  const prev = ev.count > 1 && [...logList.children].slice(-4).find((n) => n.dataset.key === key);
  if (prev) {                                           // a collapsed burst: update that line and keep it at the end
    prev.querySelector(".ts").textContent = ts;
    let c = prev.querySelector(".count");
    if (!c) { c = el("span", "count"); prev.append(c); }
    c.textContent = `  ×${ev.count}`;
    logList.append(prev);
    if (follow) logList.scrollTop = logList.scrollHeight;
    return;
  }
  const line = el("div", `ev ${ev.kind} ${ev.level || ""} ${ev.status || ""}`);
  line.dataset.key = key;
  line.append(el("span", "ts", ts));
  const who = ev.source === "dashboard" ? "▸ " : "";
  line.append(document.createTextNode(who + ev.text));
  if (ev.count > 1) line.append(el("span", "count", `  ×${ev.count}`));
  logList.append(line);
  while (logList.childElementCount > 1500) logList.firstChild.remove();
  if (follow) logList.scrollTop = logList.scrollHeight;
  else $("log-follow").classList.remove("hidden");
}

loadMeta();
pollLive();
pollLog();
