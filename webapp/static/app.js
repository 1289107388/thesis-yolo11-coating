// COATING·DET 终端逻辑：导航 / 时钟 / 实验表 / 明细 / 在线检测
const $ = (s, p) => (p || document).querySelector(s);
const $$ = (s, p) => Array.from((p || document).querySelectorAll(s));
const state = { sample: null, results: [], sortK: null, sortAsc: false, selExp: "C_cbam" };

/* ---------- 导航与时钟 ---------- */
$("#nav").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#nav button").forEach((x) => x.classList.remove("on")); b.classList.add("on");
  $$(".view").forEach((v) => v.classList.remove("on"));
  $("#v-" + b.dataset.v).classList.add("on");
  $("#vtitle").textContent = b.textContent.replace(/\s+/g, " ").trim().toUpperCase().slice(3) + " / " + b.childNodes[1].textContent.trim();
});
setInterval(() => { $("#clock").textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false }); }, 1000);

/* ---------- 实验数据 ----------
   results.json 结构：[{id,label,params_M,mean4,domains:{val|test_1|test_2|test_3:{P,R,mAP50,mAP50_95,per_class,confusion}}}] */
async function loadResults() {
  try { state.results = await (await fetch("/static/results.json")).json(); } catch { return; }
  renderTrials();
}

function renderTrials() {
  const tb = $("#trials tbody");
  const rows = state.results.slice();
  if (state.sortK) {
    rows.sort((a, b) => {
      const va = state.sortK === "id" ? a.id : val(a, state.sortK), vb = state.sortK === "id" ? b.id : val(b, state.sortK);
      return (typeof va === "string" ? va.localeCompare(vb) : va - vb) * (state.sortAsc ? 1 : -1);
    });
  }
  tb.innerHTML = rows.map((e) => {
    const d = e.domains, c = (k) => d[k] ? d[k].mAP50.toFixed(1) + `<small>${d[k].mAP50_95.toFixed(1)}</small>` : "—";
    const mean = e.mean4 != null ? e.mean4.toFixed(1) : "—";
    const best = e.id === "C_cbam";
    return `<tr data-id="${e.id}" class="${e.id === state.selExp ? "sel" : ""}">
      <td>${e.id}</td><td>${e.label}</td><td>${e.params_M ?? "—"}</td>
      <td>${c("val")}</td><td>${c("test_1")}</td><td>${c("test_2")}</td><td>${c("test_3")}</td>
      <td class="${best ? "td-hl" : ""}">${mean}</td></tr>`;
  }).join("");
  $$("#trials tbody tr").forEach((tr) => tr.addEventListener("click", () => {
    state.selExp = tr.dataset.id; renderTrials(); renderDetail(tr.dataset.id);
  }));
  renderDetail(state.selExp);
}
function val(e, k) { return k === "mean4" || k === "params_M" ? (e[k] ?? -1) : (e.domains[k] ? e.domains[k].mAP50 : -1); }

$$("#trials th[data-k]").forEach((th) => th.addEventListener("click", () => {
  const k = th.dataset.k;
  if (state.sortK === k) state.sortAsc = !state.sortAsc; else { state.sortK = k; state.sortAsc = false; }
  renderTrials();
}));

function renderDetail(id) {
  const e = state.results.find((x) => x.id === id); if (!e) return;
  $("#ed-title").textContent = `DETAIL / ${e.id} · ${e.label} · ${e.params_M ?? "?"}M params`;
  const doms = [["val", "VAL · Low-Cold"], ["test_1", "TEST_1 · Low-Warm"], ["test_2", "TEST_2 · Low-Cold"], ["test_3", "TEST_3 · High-Cold"]];
  $("#ed-body").className = "ed";
  $("#ed-body").innerHTML = doms.map(([k, name]) => {
    const d = e.domains[k]; if (!d) return `<div class="dom"><h4>${name}</h4></div>`;
    const cf = d.confusion || {};
    const rows = [
      ["P / R", `${d.P} / ${d.R}`, 0],
      ["mAP50 / 50:0.95", `${d.mAP50} / ${d.mAP50_95}`, 0],
    ];
    if (cf.defect_as_defect !== undefined) {
      rows.push(["defect→defect 命中", cf.defect_as_defect, 0]);
      rows.push(["defect→particle 漏判", cf.defect_as_particle, 1]);
      if (cf.defect_missed !== undefined) rows.push(["defect 漏检", cf.defect_missed, 1]);
      rows.push(["particle→defect 误报", cf.particle_as_defect, 1]);
      if (cf.background_as_defect !== undefined) rows.push(["背景→defect 误报", cf.background_as_defect, 1]);
    }
    const pc = d.per_class || {};
    if (pc.defect) rows.push(["per-class mAP50", `defect ${pc.defect.mAP50}${pc.particle ? " · particle " + pc.particle.mAP50 : ""}`, 0]);
    return `<div class="dom"><h4>${name}</h4>` + rows.map(([l, v, warn]) =>
      `<div class="row${warn ? " conf" : ""}"><span>${l}</span><b>${v}</b></div>`).join("") + `</div>`;
  }).join("");
}

/* ---------- 在线检测 ---------- */
async function loadModels() {
  const r = await (await fetch("/api/models")).json();
  $("#model").innerHTML = r.models.map((m) => `<option>${m}</option>`).join("");
  const box = $("#samples");
  box.innerHTML = `<table>${r.samples.map((s, i) =>
    `<tr data-ref="${s.domain}/${s.name}" class="${i === 0 ? "sel" : ""}">
       <td class="dom">${s.domain.replace("test_", "T")}</td><td>${s.name.slice(0, 22)}…</td><td>${s.boxes}box</td>
     </tr>`).join("")}</table>`;
  state.sample = r.samples[0] ? r.samples[0].domain + "/" + r.samples[0].name : null;
  $$("#samples tr").forEach((tr) => tr.addEventListener("click", () => {
    $$("#samples tr").forEach((x) => x.classList.remove("sel")); tr.classList.add("sel");
    state.sample = tr.dataset.ref; $("#upload").value = "";
  }));
}

$("#conf").addEventListener("input", (e) => {
  $("#conf-out").textContent = e.target.value;
  $("#sys-conf").textContent = e.target.value;
});

$("#run").addEventListener("click", async () => {
  const btn = $("#run"); btn.disabled = true; btn.textContent = "◌ RUNNING…";
  $("#infer").innerHTML = "INFERENCE · running…"; $("#sl").textContent = "推理中 — " + $("#model").value;
  const fd = new FormData();
  fd.append("model", $("#model").value); fd.append("conf", $("#conf").value);
  const f = $("#upload").files[0];
  if (f) fd.append("file", f); else fd.append("sample", state.sample || "");
  try {
    const r = await (await fetch("/api/detect", { method: "POST", body: fd })).json();
    if (r.error) { $("#infer").textContent = "ERROR · " + r.error; return; }
    $("#result").src = r.image;
    $("#infer").innerHTML = `<b>DONE</b> · ${r.model} · conf≥${r.conf} · ${r.elapsed_ms} ms · ${r.detections.length} objects`;
    $("#sl").textContent = `完成 — ${r.detections.length} 个目标（defect ${r.counts.defect || 0} / particle ${r.counts.particle || 0}）`;
    $("#counts").innerHTML = `
      <div class="c-def">DEFECT<b>${r.counts.defect || 0}</b></div>
      <div class="c-par">PARTICLE<b>${r.counts.particle || 0}</b></div>
      <div class="c-all">TOTAL<b>${r.detections.length}</b></div>`;
    $("#dets").innerHTML = r.detections.map((d, i) => `<tr><td>${String(i + 1).padStart(2, "0")}</td>
      <td class="cls-${d.cls === "defect" ? "d" : "p"}">${d.cls.toUpperCase()}</td>
      <td>${d.conf.toFixed(3)}</td><td>${d.xyxy.map((v) => Math.round(v)).join("·")}</td></tr>`).join("")
      || `<tr><td colspan="4" style="color:var(--mut)">无检出</td></tr>`;
  } catch (err) {
    $("#infer").textContent = "ERROR · " + err.message;
  } finally {
    btn.disabled = false; btn.textContent = "▸ RUN DETECTION";
  }
});

loadModels();
loadResults();
