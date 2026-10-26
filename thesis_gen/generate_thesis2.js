// 论文渲染器：thesis_data.json + content_a.js + content_b.js → docx
// 段落流 DSL：p/pc 开新段，t/cite 追加到当前段（支持在 plain 字符串里拆引文）
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, PageNumber, NumberFormat, AlignmentType, HeadingLevel,
  WidthType, BorderStyle, SectionType, TableOfContents, TabStopType,
} = require("docx");
const fs = require("fs");
const path = require("path");

const RAW = JSON.parse(fs.readFileSync(path.join(__dirname, "thesis_data.json"), "utf-8"));
const D = RAW.exp;
const FIGDIR = "D:\\毕业论文\\04_论文\\figures\\";
const OUT = "D:\\毕业论文\\04_论文\\毕业论文_基于YOLOv11的风机塔筒涂层缺陷检测.docx";

const M = {
  m50: (k, d) => (D[k] && D[k][d] ? (D[k][d].mAP50 * 100).toFixed(1) : "-"),
  m95: (k, d) => (D[k] && D[k][d] ? (D[k][d].mAP50_95 * 100).toFixed(1) : "-"),
  prc: (k, d, f) => (D[k] && D[k][d] ? (D[k][d][f] * 100).toFixed(1) : "-"),
  delta: (a, b, d) => ((D[a][d].mAP50 - D[b][d].mAP50) * 100).toFixed(1),
  mean4: (k) => ((D[k].val.mAP50 + D[k].test_1.mAP50 + D[k].test_2.mAP50 + D[k].test_3.mAP50) / 4 * 100).toFixed(1),
  params: (k) => (D[k] && D[k].params_M ? String(D[k].params_M) : "-"),
  j: (k, d) => (D[k] && D[k][d] ? (D[k][d].mAP50 * 100).toFixed(1) + "/" + (D[k][d].mAP50_95 * 100).toFixed(1) : "-"),
  arrow: (a, b, d) => (D[a] && D[a][d] ? (D[a][d].mAP50 * 100).toFixed(1) : "-") + "→" + (D[b] && D[b][d] ? (D[b][d].mAP50 * 100).toFixed(1) : "-"),
  p95c: (k, d) => (D[k] && D[k][d] && D[k][d].defect ? (D[k][d].defect.mAP50_95 * 100).toFixed(1) : "-"),
};

const NB = { style: BorderStyle.NONE };
const FHEI = { eastAsia: "SimHei", ascii: "Times New Roman" };
const FSONG = { eastAsia: "SimSun", ascii: "Times New Roman" };

// ---- 字数统计（CJK）----
function countCJK(items) {
  let n = 0;
  const add = (s) => { if (s) n += (String(s).match(/[\u4e00-\u9fff]/g) || []).length; };
  for (const it of items) {
    if (it.p) add(it.p);
    if (it.pen || it.kw || it.kwen) add(it.pen || it.kw || it.kwen);
    if (Array.isArray(it.pc)) it.pc.forEach(r => add(r.t));
    if (it.t) add(it.t);
    if (it.cap) add(it.cap);
    if (it.h1) add(it.h1); if (it.h2) add(it.h2);
    if (it.table) { it.table.head.forEach(add); it.table.rows.forEach(r => r.forEach(add)); }
  }
  return n;
}

// ---- 渲染 ----
function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }

function render(items) {
  const out = [];
  let runs = null; // 当前段落累积
  const flush = () => {
    if (runs && runs.length) {
      out.push(new Paragraph({
        alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 480 },
        spacing: { line: 360 }, children: runs,
      }));
    }
    runs = null;
  };
  const flushKw = (children) => out.push(new Paragraph({ spacing: { before: 240, line: 360 }, children }));
  const push = (r) => { if (!runs) runs = []; runs.push(r); };

  for (const it of items) {
    if (it.h1) { flush(); out.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: it.h1, bold: true, size: 32, font: FHEI, color: "000000" })] })); continue; }
    if (it.h2) { flush(); out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: it.h2, bold: true, size: 30, font: FHEI, color: "000000" })] })); continue; }
    if (it.p !== undefined) { flush(); push(new TextRun({ text: it.p, size: 24 })); continue; }
    if (it.pen !== undefined) { flush(); push(new TextRun({ text: it.pen, size: 24, font: { ascii: "Times New Roman" } })); continue; }
    if (it.pc !== undefined) { flush(); it.pc.forEach(r => push(new TextRun({ text: r.t, size: 24 }))); continue; }
    if (it.t !== undefined) { push(new TextRun({ text: it.t, size: 24 })); continue; }
    if (it.cite !== undefined) { push(new TextRun({ text: "[" + it.cite + "]", superScript: true, size: 21, font: { ascii: "Times New Roman" } })); continue; }
    if (it.kw) { flush(); flushKw([new TextRun({ text: it.kw.split("：")[0] + "：", bold: true, size: 24, font: FHEI }), new TextRun({ text: it.kw.split("：")[1], size: 24 })]); continue; }
    if (it.kwen) { flush(); flushKw([new TextRun({ text: "Key words: ", bold: true, size: 24, font: { ascii: "Times New Roman" } }), new TextRun({ text: it.kwen.replace("Key words: ", ""), size: 24, font: { ascii: "Times New Roman" } })]); continue; }
    if (it.fig) {
      flush();
      const p = FIGDIR + it.fig, width = it.w || 5.6, { w, h } = pngSize(p);
      out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160 }, children: [new ImageRun({ data: fs.readFileSync(p), transformation: { width: Math.round(width * 96), height: Math.round(width * 96 * h / w) }, type: "png" })] }));
      out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 }, children: [new TextRun({ text: it.cap, size: 21, font: FSONG })] }));
      continue;
    }
    if (it.cap) { flush(); out.push(new Paragraph({ keepNext: true, alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 }, children: [new TextRun({ text: it.cap, size: 21, font: FSONG })] })); continue; }
    if (it.table) {
      flush();
      const { head, rows, widths } = it.table;
      const cm = { top: 40, bottom: 40, left: 100, right: 100 };
      const cell = (txt, bold, i, isHead) => new TableCell({
        borders: isHead
          ? { bottom: { style: BorderStyle.SINGLE, size: 3, color: "000000" }, top: NB, left: NB, right: NB }
          : { top: NB, bottom: NB, left: NB, right: NB },
        margins: cm, width: { size: widths[i], type: WidthType.PERCENTAGE },
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: String(txt), bold: isHead, size: 20, font: FSONG })] })],
      });
      out.push(new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        borders: { top: { style: BorderStyle.SINGLE, size: 6, color: "000000" }, bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000" }, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB },
        rows: [
          new TableRow({ tableHeader: true, cantSplit: true, children: head.map((x, i) => cell(x, true, i, true)) }),
          ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((x, i) => cell(x, false, i, false)) })),
        ],
      }));
      continue;
    }
    if (it.fig) {
      flush();
      const p = FIGDIR + it.fig, width = it.w || 5.6, { w, h } = pngSize(p);
      out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160 }, children: [new ImageRun({ data: fs.readFileSync(p), transformation: { width: Math.round(width * 96), height: Math.round(width * 96 * h / w) }, type: "png" })] }));
      out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 }, children: [new TextRun({ text: it.cap, size: 21, font: FSONG })] }));
      continue;
    }
    if (it.formula) {
      flush();
      const runs2 = it.formula.runs.map(r => {
        if (r.it !== undefined) return new TextRun({ text: r.it, italics: true, size: 24, font: { ascii: "Times New Roman" } });
        if (r.sub !== undefined) return new TextRun({ text: r.sub, subScript: true, size: 24 });
        if (r.sup !== undefined) return new TextRun({ text: r.sup, superScript: true, size: 24 });
        return new TextRun({ text: r.t, size: 24, font: { ascii: "Times New Roman" } });
      });
      out.push(new Paragraph({
        tabStops: [{ type: TabStopType.CENTER, position: 4394 }, { type: TabStopType.RIGHT, position: 8788 }],
        children: [new TextRun({ text: "\t", size: 24 }), ...runs2, new TextRun({ text: "\t(" + it.formula.num + ")", size: 24 })],
      }));
      continue;
    }
  }
  flush();
  return out;
}

// ---- 封面（与上版一致）----
function cover() {
  const rows = [["学　　院", "【学院名称】"], ["专　　业", "【专业名称】"], ["姓　　名", "【姓名】"], ["学　　号", "【学号】"], ["指导教师", "【教师姓名】"]];
  const tbl = new Table({
    width: { size: 60, type: WidthType.PERCENTAGE }, alignment: AlignmentType.CENTER,
    borders: { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB },
    rows: rows.map(([l, v]) => new TableRow({ cantSplit: true, children: [
      new TableCell({ width: { size: 40, type: WidthType.PERCENTAGE }, borders: { bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" }, top: NB, left: NB, right: NB }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: l, size: 28, bold: true, font: FHEI })] })] }),
      new TableCell({ width: { size: 60, type: WidthType.PERCENTAGE }, borders: { bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" }, top: NB, left: NB, right: NB }, margins: { top: 60, bottom: 60, left: 120, right: 120 }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: v, size: 28, font: FSONG })] })] }),
    ] })),
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 900, after: 300, line: 506, lineRule: "atLeast" }, children: [new TextRun({ text: "【大学名称】", size: 44, bold: true, font: FHEI })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 700, line: 414, lineRule: "atLeast" }, children: [new TextRun({ text: "本科毕业论文（设计）", size: 36, bold: true, font: FHEI })] }),
    tbl,
    new Paragraph({ spacing: { before: 1100 }, children: [] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200, line: 414, lineRule: "atLeast" }, children: [new TextRun({ text: "基于 YOLOv11 的风机塔筒涂层缺陷检测", size: 36, bold: true, font: FHEI })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 700, line: 368, lineRule: "atLeast" }, children: [new TextRun({ text: "Coating Defect Detection for Wind Turbine Towers Based on YOLOv11", size: 28, italics: true, font: { ascii: "Times New Roman" } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 900, line: 322, lineRule: "atLeast" }, children: [new TextRun({ text: "2027 年 6 月", size: 28, font: FSONG })] }),
  ];
}

const refTexts = [
  "PÉREZ GARCÍA DE LA PUENTE N L, et al. CoatingDet: Coating Defect Detection Dataset (Version 2) [DS/OL]. Zenodo, 2026. https://doi.org/10.5281/zenodo.21600391.",
  "REDMON J, DIVVYA S, GIRSHICK R, et al. You Only Look Once: Unified, Real-Time Object Detection [C]// CVPR. 2016: 779-788.",
  "REN S, HE K, GIRSHICK R, et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks [J]. IEEE TPAMI, 2017, 39(6): 1137-1149.",
  "LIU W, ANGUELOV D, ERHAN D, et al. SSD: Single Shot MultiBox Detector [C]// ECCV. 2016: 21-37.",
  "REDMON J, FARHADI A. YOLOv3: An Incremental Improvement [EB/OL]. arXiv:1804.02767, 2018.",
  "BOCHKOVSKIY A, WANG C Y, LIAO H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection [EB/OL]. arXiv:2004.10934, 2020.",
  "WANG C Y, BOCHKOVSKIY A, LIAO H Y M. YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors [C]// CVPR. 2023: 7464-7475.",
  "WANG A, CHEN H, LIU L, et al. YOLOv10: Real-Time End-to-End Object Detection [EB/OL]. arXiv:2405.14458, 2024.",
  "ZHAO Y, LV W, XU S, et al. DETRs Beat YOLOs on Real-Time Object Detection [C]// CVPR. 2024: 16965-16974.",
  "KHANAM R, HUSSAIN M. YOLOv11: An Overview of the Key Architectural Enhancements [EB/OL]. arXiv:2410.17725, 2024.",
  "AKYON F C, ALTINUC S O, TEMIZEL A. Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection [C]// ICIP. 2022: 966-970.",
  "WOO S, PARK J, LEE J Y, et al. CBAM: Convolutional Block Attention Module [C]// ECCV. 2018: 3-19.",
  "HOU Q, ZHOU D, FENG J. Coordinate Attention for Efficient Mobile Network Design [C]// CVPR. 2021: 13713-13722.",
  "TONG Y, QIU Y, SONG Y, et al. Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism [EB/OL]. arXiv:2301.10051, 2023.",
  "SELVARAJU R R, COGSWELL M, DAS A, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization [J]. IJCV, 2020, 128(2): 336-359.",
  "JOCHER G, QIU J. Ultralytics YOLO (Version 8.3) [EB/OL]. 2023. https://github.com/ultralytics/ultralytics.",
  "GE Z, LIU S, WANG F, et al. YOLOX: Exceeding YOLO Series in 2021 [EB/OL]. arXiv:2107.08430, 2021.",
  "BUCHANAN S. Grey-World Colour Constancy [EB/OL]. OpenCV Documentation, 2024. https://docs.opencv.org/4.x/",
];
const refs = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "参考文献", bold: true, size: 32, font: FHEI, color: "000000" })] }),
  ...refTexts.map((t, i) => new Paragraph({
    indent: { left: 420, hanging: 420 }, spacing: { line: 360 },
    children: [new TextRun({ text: "[" + (i + 1) + "] " + t, size: 21 })],
  })),
];
const thanks = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "致　谢", bold: true, size: 32, font: FHEI, color: "000000" })] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 480 }, spacing: { line: 360 }, children: [new TextRun({ text: "本论文的完成离不开指导教师的悉心指导与宝贵建议，从选题、实验设计到论文撰写都给予了耐心指点，在此谨表诚挚的感谢。感谢 CoatingDet 数据集与官方代码的开源贡献者，公开的数据与可复现的基准使本文的实验得以在坚实的起点上展开；感谢一同学习、相互讨论的同学们，以及所有在学业上给予过我帮助的老师和家人。本人学术水平有限，文中不足之处恳请各位老师批评指正。", size: 24 })] }),
];

const A = require("./content_a.js")(D, M);
const B = require("./content_b.js")(D, M);

// 字数校验
const allItems = [...A.absCN, ...A.ch1, ...A.ch2, ...A.ch3, ...B.ch4, ...B.ch5, ...B.ch6];
const cjk = countCJK(allItems);
console.log("CJK char count (body+abstract):", cjk);
if (cjk < 20000) { console.error("BELOW 20000, aborting"); process.exit(1); }

function pageHeader() {
  return new Header({ children: [new Paragraph({ alignment: AlignmentType.CENTER, border: { bottom: { style: BorderStyle.SINGLE, size: 1, color: "000000" } }, children: [new TextRun({ text: "基于 YOLOv11 的风机塔筒涂层缺陷检测", size: 18, color: "333333", font: FSONG })] })] });
}
function pageFooter() {
  return new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "- ", size: 21 }), new TextRun({ children: [PageNumber.CURRENT], size: 21 }), new TextRun({ text: " -", size: 21 }),
  ] })] });
}
const BM = { top: 1440, bottom: 1440, left: 1701, right: 1417, header: 850, footer: 992 };

const doc = new Document({
  styles: { default: {
    document: { run: { font: FSONG, size: 24, color: "000000" }, paragraph: { spacing: { line: 360 } } },
    heading1: { run: { font: FHEI, size: 32, bold: true, color: "000000" }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 480, after: 360, line: 360 } } },
    heading2: { run: { font: FHEI, size: 30, bold: true, color: "000000" }, paragraph: { spacing: { before: 360, after: 240, line: 360 } } },
    heading3: { run: { font: FHEI, size: 28, bold: true, color: "000000" }, paragraph: { spacing: { before: 240, after: 120, line: 360 } } },
  } },
  sections: [
    { properties: { page: { margin: { top: 0, bottom: 0, left: 0, right: 0 } } }, headers: { default: new Header({ children: [] }) }, footers: { default: new Footer({ children: [] }) }, children: cover() },
    { properties: { type: SectionType.NEXT_PAGE, page: { margin: BM, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } } }, headers: { default: pageHeader() }, footers: { default: pageFooter() }, children: render(A.absCN) },
    { properties: { type: SectionType.NEXT_PAGE, page: { margin: BM, pageNumbers: { formatType: NumberFormat.UPPER_ROMAN } } }, headers: { default: pageHeader() }, footers: { default: pageFooter() }, children: render(A.absEN) },
    { properties: { type: SectionType.NEXT_PAGE, page: { margin: BM, pageNumbers: { formatType: NumberFormat.UPPER_ROMAN } } }, headers: { default: pageHeader() }, footers: { default: pageFooter() }, children: [
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "目　录", bold: true, size: 32, font: FHEI })] }),
      new TableOfContents("目录", { hyperlink: true, headingStyleRange: "1-3" }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200 }, children: [new TextRun({ text: "（打开文档后右键目录→更新域以刷新页码）", italics: true, size: 18, color: "888888" })] }),
    ] },
    { properties: { type: SectionType.NEXT_PAGE, page: { margin: BM, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } }, headers: { default: pageHeader() }, footers: { default: pageFooter() }, children: [...render(A.ch1), ...render(A.ch2), ...render(A.ch3), ...render(B.ch4), ...render(B.ch5), ...render(B.ch6)] },
    { properties: { type: SectionType.NEXT_PAGE, page: { margin: BM, pageNumbers: { formatType: NumberFormat.DECIMAL } } }, headers: { default: pageHeader() }, footers: { default: pageFooter() }, children: refs },
    { properties: { type: SectionType.NEXT_PAGE, page: { margin: BM, pageNumbers: { formatType: NumberFormat.DECIMAL } } }, headers: { default: pageHeader() }, footers: { default: pageFooter() }, children: thanks },
  ],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log("THESIS WRITTEN", OUT, buf.length, "bytes"); });
