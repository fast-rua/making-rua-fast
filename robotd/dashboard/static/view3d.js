// robotd dashboard: the live 3D view (robot model, commanded "ghost", TCP frames, point clouds). World frame, +z up.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const host = document.getElementById("three");
const msg = document.getElementById("view-msg");
const $ = (id) => document.getElementById(id);

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e12);
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 50);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const VIEWS = {
  iso: [[1.35, -1.05, 0.85], [0.3, 0, 0.12]],
  front: [[1.75, 0, 0.42], [0.3, 0, 0.15]],
  side: [[0.35, -1.75, 0.38], [0.35, 0, 0.15]],
  top: [[0.35, -0.0001, 2.0], [0.35, 0, 0]],
};
function setView(name) {
  const [p, t] = VIEWS[name];
  camera.position.set(...p); controls.target.set(...t); controls.update();
}
setView("iso");
document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));

const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x1a1e24, 0.9);
hemi.position.set(0, 0, 1);
scene.add(hemi);
const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(1.2, -0.8, 2.0);
scene.add(key);
const fill = new THREE.DirectionalLight(0x9fb8ff, 0.5);
fill.position.set(-1, 1, 0.8);
scene.add(fill);

const grid = new THREE.GridHelper(3, 30, 0x2f3944, 0x1a2028);
grid.rotation.x = Math.PI / 2;
grid.position.z = -0.001;
scene.add(grid);
scene.add(new THREE.AxesHelper(0.1));

const robot = new THREE.Group(), ghost = new THREE.Group(), clouds = new THREE.Group(), tcps = new THREE.Group();
scene.add(robot, ghost, clouds, tcps);
// the external force on each gripper: from its TCP, along the force, 1 cm per N (hidden under 1 N)
const forceArrows = {};
const forceColour = (n) => new THREE.Color().setHSL(Math.max(0, 0.12 - n / 150), 1.0, 0.55);   // orange -> red
let geomMeshes = [], ghostMeshes = [], loadedSeq = -1, loading = false;

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + "px"; renderer.domElement.style.height = h + "px";
  camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);
resize();

// ------------------------------------------------------------------ the robot model
function colourFor(g) {
  const [r, gg, b, a] = g.rgba;
  if (!(Math.abs(r - 0.5) < 1e-3 && Math.abs(gg - 0.5) < 1e-3 && Math.abs(b - 0.5) < 1e-3)) return new THREE.Color(r, gg, b);
  if (g.body.includes("tip")) return new THREE.Color(0x24292f);
  if (g.body.includes("camera")) return new THREE.Color(0x8a96a3);
  if (g.body.includes("gripper")) return new THREE.Color(0x59616b);
  if (g.body.includes("base")) return new THREE.Color(0x353c45);
  return new THREE.Color(0x454d57);
}

function primitive(g) {
  const s = g.size;
  switch (g.type) {
    case 2: return new THREE.SphereGeometry(s[0], 20, 14);
    case 3: { const c = new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 16); c.rotateX(Math.PI / 2); return c; }
    case 4: { const e = new THREE.SphereGeometry(1, 20, 14); e.scale(s[0], s[1], s[2]); return e; }
    case 5: { const c = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 20); c.rotateX(Math.PI / 2); return c; }
    case 6: return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    default: return null;
  }
}

async function loadModel(seq) {
  loading = true;
  msg.textContent = "loading the 3D model…"; msg.classList.remove("hidden");
  try {
    const j = await (await fetch("api/model", { cache: "no-store" })).json();
    const model = j.model;
    if (!model || model.error) { msg.textContent = model?.error || "no model"; msg.classList.remove("hidden"); return; }
    const blob = await (await fetch("api/mesh.bin", { cache: "no-store" })).arrayBuffer();
    for (const grp of [robot, ghost]) { grp.children.slice().forEach((c) => { grp.remove(c); }); }
    const meshGeoms = model.meshes.map((m) => {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(blob, m.vert_offset, m.vert_count * 3), 3));
      geo.setIndex(new THREE.BufferAttribute(new Uint32Array(blob, m.face_offset, m.face_count * 3), 1));
      geo.computeVertexNormals();
      return geo;
    });
    const ghostMat = new THREE.MeshBasicMaterial({ color: 0x4cc3ff, transparent: true, opacity: 0.2, depthWrite: false });
    geomMeshes = []; ghostMeshes = [];
    for (const g of model.geoms) {
      const geo = g.mesh >= 0 ? meshGeoms[g.mesh] : primitive(g);
      if (!geo) { geomMeshes.push(null); ghostMeshes.push(null); continue; }
      const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: colourFor(g), metalness: 0.25, roughness: 0.55 }));
      mesh.matrixAutoUpdate = false;
      robot.add(mesh); geomMeshes.push(mesh);
      const gm = new THREE.Mesh(geo, ghostMat);
      gm.matrixAutoUpdate = false;
      ghost.add(gm); ghostMeshes.push(gm);
    }
    if (model.table) {
      const [sx, sy, sz] = model.table.size;
      const table = new THREE.Mesh(new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz),
        new THREE.MeshStandardMaterial({ color: 0x6b5a48, transparent: true, opacity: 0.22, depthWrite: false }));
      table.position.set(...model.table.pos);
      robot.add(table);
    }
    tcps.clear();
    for (const _ of [0, 1]) tcps.add(new THREE.AxesHelper(0.06));
    tcps.children.forEach((a) => { a.matrixAutoUpdate = false; });
    loadedSeq = seq;
    msg.classList.add("hidden");
  } catch (e) {
    msg.textContent = "3D model failed to load: " + e; msg.classList.remove("hidden");
  } finally { loading = false; }
}

const M = new THREE.Matrix4();
function applyPoses(meshes, poses) {
  if (!poses) return;
  for (let i = 0; i < meshes.length; i++) {
    const mesh = meshes[i];
    if (!mesh) continue;
    const o = i * 12, p = poses;
    M.set(p[o + 3], p[o + 4], p[o + 5], p[o], p[o + 6], p[o + 7], p[o + 8], p[o + 1], p[o + 9], p[o + 10], p[o + 11], p[o + 2], 0, 0, 0, 1);
    mesh.matrix.copy(M);
  }
}

// ------------------------------------------------------------------ point clouds
let cloudSeq = -1, cloudBusy = false, cloudParts = [];
const pointMat = () => new THREE.PointsMaterial({ size: Number($("point-size").value), sizeAttenuation: false, vertexColors: true });
const camVisible = {};

function heightColours(pos) {
  const n = pos.length / 3, out = new Float32Array(n * 3), c = new THREE.Color();
  for (let i = 0; i < n; i++) {
    const z = pos[3 * i + 2], t = Math.max(0, Math.min(1, (z + 0.02) / 0.35));
    c.setHSL(0.66 - 0.66 * t, 0.85, 0.55);
    out[3 * i] = c.r; out[3 * i + 1] = c.g; out[3 * i + 2] = c.b;
  }
  return out;
}

function buildCloudToggles(names) {
  const box = $("cloud-cams");
  const have = [...box.querySelectorAll("input")].map((i) => i.dataset.cam).join(",");
  if (have === names.join(",")) return;
  box.innerHTML = "";
  for (const n of names) {
    if (camVisible[n] === undefined) camVisible[n] = true;
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" data-cam="${n}" ${camVisible[n] ? "checked" : ""}> ${n}`;
    l.querySelector("input").addEventListener("change", (e) => { camVisible[n] = e.target.checked; updateCloudVisibility(); });
    box.append(l);
  }
}

function updateCloudVisibility() {
  clouds.visible = $("show-cloud").checked;
  for (const p of cloudParts) p.visible = camVisible[p.userData.cam] !== false;
}

async function loadCloud(seq) {
  cloudBusy = true;
  try {
    const buf = await (await fetch("api/cloud.bin", { cache: "no-store" })).arrayBuffer();
    if (buf.byteLength < 4) return;
    const headLen = new DataView(buf).getUint32(0, true);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headLen)));
    const total = head.counts.reduce((s, c) => s + c[1], 0);
    const P = new Float32Array(buf, 4 + headLen, total * 3);
    const C = new Uint8Array(buf, 4 + headLen + total * 12, total * 3);
    for (const p of cloudParts) { clouds.remove(p); p.geometry.dispose(); p.material.dispose(); }
    cloudParts = [];
    let o = 0;
    for (const [name, n] of head.counts) {
      const geo = new THREE.BufferGeometry();
      const pos = P.subarray(o * 3, (o + n) * 3);
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("color", $("color-height").checked
        ? new THREE.BufferAttribute(heightColours(pos), 3)
        : new THREE.BufferAttribute(C.subarray(o * 3, (o + n) * 3), 3, true));
      const pts = new THREE.Points(geo, pointMat());
      pts.userData.cam = name;
      clouds.add(pts); cloudParts.push(pts);
      o += n;
    }
    buildCloudToggles(head.counts.map((c) => c[0]));
    updateCloudVisibility();
    cloudSeq = seq;
  } catch (e) { /* next sequence number retries */ } finally { cloudBusy = false; }
}

$("show-robot").addEventListener("change", (e) => { robot.visible = e.target.checked; });
$("show-ghost").addEventListener("change", () => {});
$("show-cloud").addEventListener("change", updateCloudVisibility);
$("point-size").addEventListener("input", (e) => { for (const p of cloudParts) p.material.size = Number(e.target.value); });
$("color-height").addEventListener("change", () => { cloudSeq = -1; });

// ------------------------------------------------------------------ live updates
let latest = null, applied = null;
window.addEventListener("robot-live", (e) => { latest = e.detail; });

function frame() {
  requestAnimationFrame(frame);
  const live = latest;
  if (live && live !== applied) {
    applied = live;
    if (live.connected) {
      if (live.meta_seq !== loadedSeq && !loading && live.meta_seq > 0) loadModel(live.meta_seq);
      applyPoses(geomMeshes, live.geoms);
      ghost.visible = $("show-ghost").checked && !!live.ghost;
      if (live.ghost) applyPoses(ghostMeshes, live.ghost);
      const arms = Object.keys(live.ee || {});
      tcps.children.forEach((ax, i) => {
        const e = live.ee?.[arms[i]];
        if (!e) return;
        const T = e.T;
        ax.matrix.set(T[0][0], T[0][1], T[0][2], T[0][3], T[1][0], T[1][1], T[1][2], T[1][3],
          T[2][0], T[2][1], T[2][2], T[2][3], 0, 0, 0, 1);
      });
      if (live.cloud_seq !== cloudSeq && !cloudBusy && live.cloud_seq > 0) loadCloud(live.cloud_seq);
      const passive = window.__meta?.passive;
      for (const [arm, e] of Object.entries(live.ee || {})) {
        if (!forceArrows[arm]) {
          forceArrows[arm] = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.1, 0xff8a3d, 0.025, 0.014);
          scene.add(forceArrows[arm]);
        }
        const a = forceArrows[arm], F = new THREE.Vector3(-e.wrench[0], -e.wrench[1], -e.wrench[2]);  // on the tool
        const n = F.length();
        a.visible = $("show-force").checked && !passive && n >= 1.0;
        if (a.visible) {
          a.position.set(e.T[0][3], e.T[1][3], e.T[2][3]);
          a.setDirection(F.divideScalar(n));
          a.setLength(Math.min(0.01 * n, 0.3) + 0.03, 0.025, 0.014);
          a.setColor(forceColour(n));
        }
      }
    }
  }
  controls.update();
  renderer.render(scene, camera);
}
msg.textContent = "waiting for the driver…";
frame();
