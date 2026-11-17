// robotd dashboard: the Programs tab: list, edit, run and stop programs; their output.
"use strict";

(() => {
  let programs = [], selected = null, dirty = false, outSeq = 0, status = null, delArmed = null;
  const src = $("prog-src"), msg = $("prog-msg"), out = $("prog-out");

  const post = async (verb, body) => {
    const r = await fetch("api/programs/" + verb, {
      method: "POST", headers: { "X-Dashboard": "1", "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
    });
    return r.json();
  };
  const say = (text, cls) => { msg.textContent = text || ""; msg.className = "prog-msg " + (cls || "dim"); };
  const isMine = () => selected && selected.origin === "mine";
  const running = () => status && status.running;

  // ---------------------------------------------------------------- list
  async function refresh() {
    try {
      const j = await (await fetch("api/programs", { cache: "no-store" })).json();
      if (!j.ok) { const d = document.createElement("div"); d.className = "prog-item"; const sp = document.createElement("span"); sp.className = "pd"; sp.textContent = j.error; d.append(sp); $("prog-list").replaceChildren(d); return; }
      programs = j.programs;
      $("prog-new-name").title = "a new program in " + j.dir;
      render();
    } catch (e) { /* the live poll reports outages */ }
  }

  function render() {
    const box = $("prog-list");
    box.innerHTML = "";
    const run = running();
    for (const p of programs) {
      const row = el("div", "prog-item");
      if (selected && p.name === selected.name && p.origin === selected.origin) row.classList.add("sel");
      if (run && run.name === p.name && run.origin === p.origin) row.classList.add("running");
      row.append(el("span", "pn", p.name), el("span", "pd", p.doc || ""), el("span", "po", p.origin === "mine" ? "mine" : "example"));
      row.title = p.doc || p.name;
      row.addEventListener("click", () => open(p.name, p.origin));
      box.append(row);
    }
    if (!programs.length) box.append(el("div", "prog-item", "no programs yet: make one with New, or copy an example"));
  }

  async function open(name, origin) {
    if (dirty && selected && !(selected.name === name && selected.origin === origin)) {
      if (!open.warned) { say(`${selected.name} has unsaved changes: Save, or click again to discard them`, "err"); open.warned = true; return; }
    }
    open.warned = false;
    const j = await (await fetch(`api/programs/source?name=${encodeURIComponent(name)}&origin=${origin}`, { cache: "no-store" })).json();
    if (!j.ok) { say(j.error, "err"); return; }
    selected = { name, origin };
    src.value = j.source;
    src.readOnly = origin !== "mine";
    dirty = false;
    $("prog-name").textContent = name;
    const o = $("prog-origin");
    o.textContent = origin === "mine" ? "mine" : "example · read-only"; o.classList.remove("hidden");
    say(origin === "mine" ? "" : "an example: run it as it is, or Copy to mine to edit it");
    controls();
    render();
  }

  function controls() {
    const run = running();
    $("prog-run").disabled = !selected || !!run;
    $("prog-save").disabled = !isMine() || !dirty;
    $("prog-copy").disabled = !selected || isMine();
    $("prog-del").disabled = !isMine() || !!(run && run.name === selected.name && run.origin === "mine");
    $("prog-stop").disabled = !run || !!run.stopping;
  }

  // ---------------------------------------------------------------- edit
  src.addEventListener("input", () => { if (isMine()) { dirty = true; controls(); } });
  src.addEventListener("keydown", (e) => {
    if (e.key === "Tab") {                                       // indent, do not leave the editor
      e.preventDefault();
      const s = src.selectionStart;
      src.setRangeText("    ", s, src.selectionEnd, "end");
      src.dispatchEvent(new Event("input"));
    } else if ((e.metaKey || e.ctrlKey) && e.key === "s") {
      e.preventDefault(); save();
    }
  });

  async function save() {
    if (!isMine()) return false;
    const j = await post("save", { name: selected.name, source: src.value });
    if (!j.ok) { say(j.error, "err"); return false; }
    dirty = false;
    say(j.syntax ? `saved, but a syntax error: ${j.syntax}` : "saved", j.syntax ? "err" : "ok");
    controls(); refresh();
    return true;
  }
  $("prog-save").addEventListener("click", save);

  $("prog-copy").addEventListener("click", async () => {
    const j = await post("copy", { name: selected.name, origin: selected.origin });
    if (!j.ok) { say(j.error, "err"); return; }
    await refresh(); open(j.name, "mine"); say(`copied to your programs as ${j.name}`, "ok");
  });

  $("prog-del").addEventListener("click", async () => {
    if (delArmed !== selected.name) {
      delArmed = selected.name; say(`click Delete again to delete ${selected.name}`, "err");
      setTimeout(() => { delArmed = null; }, 3000);
      return;
    }
    delArmed = null;
    const j = await post("delete", { name: selected.name });
    if (!j.ok) { say(j.error, "err"); return; }
    say(`deleted ${selected.name}`, "ok");
    selected = null; src.value = ""; dirty = false; $("prog-name").textContent = "pick a program";
    $("prog-origin").classList.add("hidden");
    controls(); refresh();
  });

  $("prog-new").addEventListener("click", async () => {
    let name = $("prog-new-name").value.trim();
    if (!name) { say("type a name for the new program first", "err"); return; }
    if (!name.endsWith(".py")) name += ".py";
    const j = await post("save", { name });
    if (!j.ok) { say(j.error, "err"); return; }
    $("prog-new-name").value = "";
    await refresh(); open(name, "mine"); say(`made ${name} from the template`, "ok");
  });

  // ---------------------------------------------------------------- run / stop
  $("prog-run").addEventListener("click", async () => {
    if (dirty && !(await save())) return;
    const j = await post("run", { name: selected.name, origin: selected.origin, args: $("prog-args").value });
    if (!j.ok) { say(j.error, "err"); return; }
    say(`started ${selected.name}`, "ok");
  });
  $("prog-stop").addEventListener("click", async () => {
    const j = await post("stop", {});
    say(j.ok ? (j.stopped ? "stopping…" : j.why) : j.error, j.ok ? "dim" : "err");
  });
  $("prog-clear").addEventListener("click", () => { out.innerHTML = ""; });

  async function pollOutput() {
    try {
      const j = await (await fetch("api/programs/output?since=" + outSeq, { cache: "no-store" })).json();
      if (j.ok && j.lines.length) {
        const follow = out.scrollTop + out.clientHeight >= out.scrollHeight - 8;
        for (const l of j.lines) {
          const line = el("div", l.stream === "err" ? "err" : l.stream === "system" ? "system" : "");
          line.textContent = l.text;
          out.append(line);
        }
        while (out.childElementCount > 3000) out.firstChild.remove();
        if (follow) out.scrollTop = out.scrollHeight;
      }
      if (j.ok) outSeq = j.seq;
    } catch (e) { /* retried */ }
    setTimeout(pollOutput, 300);
  }

  // ---------------------------------------------------------------- status from the live poll
  window.addEventListener("robot-live", (e) => {
    const was = running() && running().name;
    status = e.detail.program;
    const run = running(), last = status && status.last;
    const chip = $("top-prog"), st = $("prog-status"), dot = $("prog-dot");
    if (run) {
      const label = `▶ ${run.name} ${fmt(run.runtime, 0)} s` + (run.stopping ? " (stopping)" : "");
      chip.textContent = label; chip.classList.remove("hidden");
      st.textContent = label + (run.args ? `  ${run.args}` : ""); st.className = "";
      dot.style.background = "var(--ok)";
    } else {
      chip.classList.add("hidden");
      st.textContent = last ? `last: ${last.name} ${last.stopped_by ? "stopped (" + last.stopped_by + ")" : "exited " + last.code} after ${fmt(last.runtime, 1)} s` : "no program running";
      st.className = "dim";
      dot.style.background = "transparent";
    }
    if ((run && run.name) !== was) render();
    controls();
  });

  $("tabbar").querySelector('[data-tab="programs"]').addEventListener("click", () => { Joints.select("programs"); refresh(); });
  $("side-wide").addEventListener("click", () => $("app").classList.toggle("wide"));
  refresh();
  pollOutput();
})();
