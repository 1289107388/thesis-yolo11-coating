// 毕业设计演示页逻辑：选项卡 / 结果图表 / 在线检测
const $ = (s) => document.querySelector(s);
const state = { sample: null };

document.querySelectorAll(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("#" + b.dataset.tab).classList.add("active");
  })
);

async function loadModels() {
  const r = await (await fetch("/api/models")).json();
  $("#model").innerHTML = r.models.map((m) => `<option>${m}</option>`).join("");
  const box = $("#samples");
  box.innerHTML = r.samples
    .map(
      (s, i) =>
        `<img src="${s.url}" data-ref="${s.domain}/${s.name}" title="${s.domain} · ${s.boxes} 框" class="${i === 0 ? "sel" : ""}">`
    )
    .join("");
  state.sample = r.samples[0] ? r.samples[0].domain + "/" + r.samples[0].name : null;
  box.querySelectorAll("img").forEach((im) =>
    im.addEventListener("click", () => {
      box.querySelectorAll("img").forEach((x) => x.classList.remove("sel"));
      im.classList.add("sel");
      state.sample = im.dataset.ref;
      $("#upload").value = "";
    })
  );
}

$("#conf").addEventListener("input", (e) => ($("#conf-out").textContent = e.target.value));

$("#run").addEventListener("click", async () => {
  const btn = $("#run");
  btn.disabled = true;
  btn.textContent = "检测中…";
  const fd = new FormData();
  fd.append("model", $("#model").value);
  fd.append("conf", $("#conf").value);
  const file = $("#upload").files[0];
  if (file) fd.append("file", file);
  else fd.append("sample", state.sample || "");
  try {
    const r = await (await fetch("/api/detect", { method: "POST", body: fd })).json();
    if (r.error) { alert(r.error); return; }
    $("#result").src = r.image;
    const c = r.counts;
    $("#counts").innerHTML = `模型 <b>${r.model}</b> @conf ${r.conf}：检出 <b>${r.detections.length}</b> 个目标` +
      (c.defect ? `，其中 <b>defect（需返修）${c.defect}</b>` : "") +
      (c.particle ? `，particle（良性灰尘）${c.particle}` : "");
    $("#dets").textContent = r.detections
      .map((d) => `${d.cls} conf=${d.conf} [${d.xyxy.map((v) => v.toFixed(0)).join(", ")}]`)
      .join("\n");
  } finally {
    btn.disabled = false;
    btn.textContent = "开始检测";
  }
});

// ---- 实验结果表与条形图 ----
const EXP_LABEL = {
  A0: "A0 基线 yolo11n", B_p2: "B +P2 检测头", C_cbam: "C +CBAM", D_wiou: "D +WIoUv3",
  E_gw: "E +白平衡&宽色相", F_combo: "F 全组合", A0_60ep: "A0 基线@60ep",
  F2_C_E: "F2 = C+E", F3_C_D_E: "F3 = C+D+E", D2_wiou_rclamp: "D2 WIoU+下限截断",
  v5s: "YOLOv5s", v8s: "YOLOv8s", v10s: "YOLOv10s",
};
async function loadResults() {
  let data;
  try { data = await (await fetch("/static/results.json")).json(); } catch { return; }
  const tb = $("#res-table tbody");
  tb.innerHTML = Object.entries(data)
    .filter(([, v]) => v && v.val)
    .map(([k, v]) => {
      const cell = (d) => {
        const m = v[d] && v[d].overall;
        return m ? `${(m.mAP50 * 100).toFixed(1)}<small>${(m.mAP50_95 * 100).toFixed(1)}</small>` : "-";
      };
      return `<tr><td>${EXP_LABEL[k] || k}</td><td>${cell("val")}</td><td>${cell("test_1")}</td><td>${cell("test_2")}</td><td>${cell("test_3")}</td></tr>`;
    })
    .join("");
  drawBars(data);
  $("#metric-domain").addEventListener("change", () => drawBars(data));
}

function drawBars(data) {
  const domain = $("#metric-domain").value;
  const rows = Object.entries(data)
    .filter(([, v]) => v && v[domain] && v[domain].overall)
    .map(([k, v]) => ({ name: EXP_LABEL[k] || k, val: v[domain].overall.mAP50 }))
    .sort((a, b) => b.val - a.val);
  const best = rows.length ? rows[0].val : 1;
  $("#bars").innerHTML = rows
    .map(
      (r) => `<div class="bar-row"><div class="bar-name">${r.name}</div>
        <div class="bar ${r.val === best ? "best" : ""}" style="width:${(r.val / best) * 74}%"></div>
        <div class="bar-val">${(r.val * 100).toFixed(1)}</div></div>`
    )
    .join("");
}

loadModels();
loadResults();
