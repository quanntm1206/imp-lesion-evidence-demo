import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(process.env.PROJECT_ROOT || path.join(HERE, "..", ".."));
const WORKSPACE = path.resolve(
  process.env.ARTIFACT_TOOL_WORKSPACE ||
    path.join(os.tmpdir(), "codex-presentations", "imp-evidence-deck", "tmp"),
);
const OUTPUT = path.resolve(
  process.env.OUTPUT_PPTX ||
    path.join(ROOT, "outputs", "imp-lesion-evidence-defense.pptx"),
);
const QA_DIR = path.resolve(
  process.env.PRESENTATION_QA_DIR || path.join(WORKSPACE, "qa"),
);

const artifactEntry = path.join(
  WORKSPACE,
  "node_modules",
  "@oai",
  "artifact-tool",
  "dist",
  "artifact_tool.mjs",
);
const { Presentation, PresentationFile } = await import(
  pathToFileURL(artifactEntry).href
);

const content = JSON.parse(
  await fs.readFile(path.join(ROOT, "presentation", "interactive", "content.json"), "utf8"),
);
const assetDir = path.join(ROOT, "presentation", "interactive", "assets");

const C = {
  ivory: "#F4EFE4",
  paper: "#FFFDF7",
  graphite: "#1D211F",
  muted: "#66706B",
  teal: "#177D76",
  tealSoft: "#D6E7E1",
  rust: "#B54E36",
  rustSoft: "#EAD2C9",
  sand: "#D8C39D",
  line: "#C9C2B5",
  white: "#FFFFFF",
};

const W = 1280;
const H = 720;
const FRAME = { left: 68, top: 48, width: 1144, height: 624 };
const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function addBox(slide, name, left, top, width, height, fill, options = {}) {
  return slide.shapes.add({
    geometry: options.geometry || "rect",
    name,
    position: { left, top, width, height },
    fill,
    line: {
      style: "solid",
      fill: options.lineFill || "none",
      width: options.lineWidth || 0,
    },
    ...(options.borderRadius ? { borderRadius: options.borderRadius } : {}),
    ...(options.rotation ? { rotation: options.rotation } : {}),
  });
}

function addText(slide, name, value, left, top, width, height, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    fontFamily: options.fontFamily || "Trebuchet MS",
    fontSize: options.fontSize || 20,
    color: options.color || C.graphite,
    bold: options.bold || false,
    italic: options.italic || false,
    alignment: options.alignment || "left",
  };
  return shape;
}

function addChrome(slide, data, index, accent = C.teal, options = {}) {
  slide.background.fill = C.ivory;
  addBox(slide, `top-rule-${index}`, 0, 0, W, 8, accent);
  addText(slide, `kicker-${index}`, data.kicker, FRAME.left, 36, 700, 28, {
    fontSize: 12,
    bold: true,
    color: accent,
  });
  addText(slide, `slide-title-${index}`, data.title, FRAME.left, 70, 1080, options.titleHeight || 74, {
    fontFamily: "Georgia",
    fontSize: options.titleFontSize || 39,
    bold: true,
  });
  addBox(slide, `footer-rule-${index}`, FRAME.left, 652, FRAME.width, 1, C.line);
  addText(slide, `evidence-${index}`, data.evidence_label, FRAME.left, 660, 950, 22, {
    fontSize: 12,
    color: C.muted,
  });
  addText(slide, `page-${index}`, String(index).padStart(2, "0"), 1144, 660, 68, 22, {
    fontFamily: "Georgia",
    fontSize: 14,
    bold: true,
    color: accent,
    alignment: "right",
  });
  slide.speakerNotes.textFrame.setText(data.notes);
}

function addClaim(slide, data, top = 152, width = 1090, accent = C.teal) {
  addBox(slide, `${data.id}-claim-mark`, FRAME.left, top + 3, 6, 78, accent);
  addText(slide, `${data.id}-claim`, data.claim, FRAME.left + 22, top, width, 90, {
    fontFamily: "Georgia",
    fontSize: 24,
    bold: true,
    color: C.graphite,
  });
}

function addBulletLines(slide, data, left, top, width, rowHeight = 42, accent = C.teal) {
  data.body.forEach((line, i) => {
    addBox(slide, `${data.id}-bullet-${i}`, left, top + i * rowHeight + 9, 8, 8, accent, {
      geometry: "ellipse",
    });
    addText(slide, `${data.id}-body-${i}`, line, left + 22, top + i * rowHeight, width - 22, rowHeight, {
      fontSize: 18,
      color: C.graphite,
    });
  });
}

function addMetric(slide, data, left, top, width, label, value, accent = C.teal) {
  addText(slide, `${data.id}-${label}-value`, value, left, top, width, 64, {
    fontFamily: "Georgia",
    fontSize: 45,
    bold: true,
    color: accent,
  });
  addText(slide, `${data.id}-${label}-label`, label, left, top + 58, width, 42, {
    fontSize: 15,
    bold: true,
    color: C.muted,
  });
}

function addTitleSlide(data) {
  const slide = presentation.slides.add();
  slide.background.fill = C.graphite;
  addBox(slide, "title-rust-field", 0, 0, 24, H, C.rust);
  addBox(slide, "title-teal-field", 1030, 0, 250, H, C.teal);
  addText(slide, "title-kicker", data.kicker, 80, 62, 800, 28, {
    fontSize: 13,
    bold: true,
    color: C.sand,
  });
  addText(slide, "title-main", data.title, 80, 136, 840, 170, {
    fontFamily: "Georgia",
    fontSize: 64,
    bold: true,
    color: C.ivory,
  });
  addText(slide, "title-claim", data.claim, 84, 330, 780, 110, {
    fontSize: 23,
    color: C.tealSoft,
  });
  addBox(slide, "title-rule", 84, 486, 760, 2, C.rust);
  addText(slide, "title-baseline", data.body[0], 84, 510, 790, 42, {
    fontSize: 20,
    bold: true,
    color: C.ivory,
  });
  addText(slide, "title-boundary", `${data.body[1]}\n${data.body[2]}`, 84, 558, 790, 68, {
    fontSize: 16,
    color: C.sand,
  });
  addText(slide, "title-atlas", "EVIDENCE\nATLAS", 1050, 126, 180, 210, {
    fontFamily: "Georgia",
    fontSize: 37,
    bold: true,
    color: C.ivory,
    alignment: "center",
  });
  addText(slide, "title-meta", data.evidence_label, 1050, 590, 180, 48, {
    fontSize: 13,
    color: C.ivory,
    alignment: "center",
  });
  slide.speakerNotes.textFrame.setText(data.notes);
}

function addLeakageSlide(data, index) {
  const slide = presentation.slides.add();
  addChrome(slide, data, index, C.rust);
  addClaim(slide, data, 150, 1070, C.rust);
  addBox(slide, "leakage-boundary", 410, 278, 4, 280, C.graphite);
  addBox(slide, "leakage-cross-1", 364, 338, 100, 5, C.rust, { rotation: -18 });
  addBox(slide, "leakage-cross-2", 364, 424, 100, 5, C.rust, { rotation: 18 });
  addBox(slide, "leakage-cross-3", 364, 510, 100, 5, C.rust, { rotation: -18 });
  addMetric(slide, data, 82, 292, 260, "crossing patient IDs", "3", C.rust);
  addMetric(slide, data, 82, 438, 260, "cross-boundary rows", "13", C.rust);
  addBulletLines(slide, data, 520, 292, 660, 74, C.rust);
}

function addQuestionsSlide(data, index) {
  const slide = presentation.slides.add();
  addChrome(slide, data, index);
  addClaim(slide, data, 150, 1070);
  addBox(slide, "question-divider", 636, 286, 2, 280, C.line);
  addText(slide, "rq1-label", "RQ1", 76, 292, 110, 50, {
    fontFamily: "Georgia",
    fontSize: 38,
    bold: true,
    color: C.teal,
  });
  addText(slide, "rq1-question", data.body[0].replace("RQ1: ", ""), 76, 354, 500, 150, {
    fontSize: 22,
    bold: true,
  });
  addText(slide, "rq1-contract", "System comparison\nSingle recorded run\nDescriptive inference", 76, 514, 500, 90, {
    fontSize: 16,
    color: C.muted,
  });
  addText(slide, "rq2-label", "RQ2", 686, 292, 110, 50, {
    fontFamily: "Georgia",
    fontSize: 38,
    bold: true,
    color: C.rust,
  });
  addText(slide, "rq2-question", data.body[1].replace("RQ2: ", ""), 686, 354, 500, 150, {
    fontSize: 22,
    bold: true,
  });
  addText(slide, "rq2-contract", "Matched intervention\nThree selected seeds\nConditional paired inference", 686, 514, 500, 90, {
    fontSize: 16,
    color: C.muted,
  });
}

function addPipelineSlide(data, index) {
  const slide = presentation.slides.add();
  addChrome(slide, data, index);
  addClaim(slide, data, 150, 1070);
  const nodes = content.pipeline;
  const left = 70;
  const nodeW = 164;
  const gap = 26;
  const top = 340;
  for (let i = 0; i < nodes.length - 1; i += 1) {
    addBox(slide, `pipeline-edge-${i}`, left + nodeW + i * (nodeW + gap), top + 54, gap, 3, C.line);
  }
  nodes.forEach((node, i) => {
    const x = left + i * (nodeW + gap);
    const fill = node.tone === "rust" ? C.rustSoft : node.tone === "graphite" ? C.graphite : node.tone === "sand" ? C.sand : C.tealSoft;
    const color = node.tone === "graphite" ? C.ivory : C.graphite;
    addBox(slide, `pipeline-node-${i}`, x, top, nodeW, 112, fill, {
      geometry: "roundRect",
      borderRadius: "rounded-xl",
      lineFill: node.tone === "rust" ? C.rust : C.teal,
      lineWidth: 1,
    });
    addText(slide, `pipeline-number-${i}`, String(i + 1).padStart(2, "0"), x + 14, top + 14, 40, 24, {
      fontFamily: "Georgia",
      fontSize: 14,
      bold: true,
      color: node.tone === "rust" ? C.rust : node.tone === "graphite" ? C.sand : C.teal,
    });
    addText(slide, `pipeline-label-${i}`, node.label, x + 14, top + 46, nodeW - 28, 52, {
      fontSize: 16,
      bold: true,
      color,
    });
  });
  addText(slide, "pipeline-html-note", "Interactive navigation and smooth transitions are available in the offline HTML edition.", 72, 496, 1120, 50, {
    fontSize: 18,
    color: C.muted,
    alignment: "center",
  });
}

function addDataSlide(data, index) {
  const slide = presentation.slides.add();
  addChrome(slide, data, index);
  addClaim(slide, data, 150, 1070);
  addText(slide, "data-total", "2,869", 74, 294, 300, 88, {
    fontFamily: "Georgia",
    fontSize: 68,
    bold: true,
    color: C.teal,
  });
  addText(slide, "data-total-label", "leakage-audited images", 78, 380, 300, 32, {
    fontSize: 18,
    bold: true,
    color: C.muted,
  });
  const barX = 446;
  const barY = 318;
  const barW = 716;
  const trainW = Math.round((2008 / 2869) * barW);
  const valW = Math.round((431 / 2869) * barW);
  addBox(slide, "data-train-bar", barX, barY, trainW, 70, C.teal);
  addBox(slide, "data-val-bar", barX + trainW, barY, valW, 70, C.sand);
  addBox(slide, "data-test-bar", barX + trainW + valW, barY, barW - trainW - valW, 70, C.rust);
  addText(slide, "data-train-label", "TRAIN\n2,008", barX, 402, trainW, 54, { fontSize: 16, bold: true, color: C.teal });
  addText(slide, "data-val-label", "ADAPTIVE VALIDATION\n431", barX + trainW, 402, valW, 54, { fontSize: 14, bold: true, color: C.muted, alignment: "center" });
  addText(slide, "data-test-label", "SEALED TEST\n430", barX + trainW + valW, 402, barW - trainW - valW, 54, { fontSize: 14, bold: true, color: C.rust, alignment: "right" });
  addText(slide, "data-audit-chain", "IDENTITY  /  EXACT DUPLICATE  /  PERCEPTUAL DUPLICATE  /  SPLIT GROUP", 446, 500, 716, 38, {
    fontSize: 15,
    bold: true,
    color: C.graphite,
    alignment: "center",
  });
  addText(slide, "data-boundary", "Validation was opened adaptively. The 430-row test partition remains sealed.", 446, 548, 716, 48, {
    fontSize: 18,
    color: C.rust,
    alignment: "center",
  });
}

function addModelsSlide(data, index) {
  const slide = presentation.slides.add();
  addChrome(slide, data, index, C.teal, { titleFontSize: 35, titleHeight: 98 });
  addClaim(slide, data, 178, 1070);
  addBox(slide, "models-divider", 638, 324, 2, 250, C.line);
  addText(slide, "models-mit-name", "MiT-B3 U-Net", 74, 318, 500, 52, {
    fontFamily: "Georgia",
    fontSize: 34,
    bold: true,
    color: C.teal,
  });
  addText(slide, "models-mit-spec", "384 × 384\nLAB-luminance CLAHE\nPercentile stretch + median filter\nFixed preprocessing-aware system", 74, 388, 510, 158, {
    fontSize: 20,
  });
  addText(slide, "models-nnunet-name", "nnU-Net v2", 690, 318, 500, 52, {
    fontFamily: "Georgia",
    fontSize: 34,
    bold: true,
    color: C.rust,
  });
  addText(slide, "models-nnunet-spec", "256 × 256\nRaw RGB\nGenerated plans\nSelf-configuring 2D system", 690, 388, 510, 158, {
    fontSize: 20,
  });
  addBox(slide, "models-caution", 74, 568, 1110, 42, C.graphite);
  addText(slide, "models-caution-text", "Different resolution, decoder, loss, augmentation, and policy: this is not an isolated encoder ablation.", 92, 576, 1076, 28, {
    fontSize: 16,
    bold: true,
    color: C.ivory,
    alignment: "center",
  });
}

function addValidationSlide(data, index) {
  const slide = presentation.slides.add();
  addChrome(slide, data, index);
  addClaim(slide, data, 150, 1070);
  const rows = [
    { name: "Robust Dice", a: 0.8959, b: 0.9019, y: 320 },
    { name: "Boundary F1", a: 0.4145, b: 0.4369, y: 466 },
  ];
  rows.forEach((row, i) => {
    addText(slide, `validation-label-${i}`, row.name, 76, row.y - 10, 210, 30, { fontSize: 19, bold: true });
    addText(slide, `validation-a-value-${i}`, row.a.toFixed(4), 302, row.y - 10, 95, 30, { fontSize: 18, bold: true, color: C.teal, alignment: "right" });
    addBox(slide, `validation-a-bar-${i}`, 416, row.y, Math.round(row.a * 520), 24, C.teal);
    addText(slide, `validation-b-value-${i}`, row.b.toFixed(4), 302, row.y + 34, 95, 30, { fontSize: 18, bold: true, color: C.rust, alignment: "right" });
    addBox(slide, `validation-b-bar-${i}`, 416, row.y + 44, Math.round(row.b * 520), 24, C.rust);
  });
  addText(slide, "validation-legend", "MiT-B3     nnU-Net v2", 76, 594, 420, 30, { fontSize: 16, bold: true, color: C.muted });
  addText(slide, "validation-limit", "Point estimates only\nSingle recorded run\nNo paired confidence interval", 988, 318, 210, 138, {
    fontFamily: "Georgia",
    fontSize: 21,
    bold: true,
    color: C.rust,
  });
  addText(slide, "validation-precision", "MiT-B3 retains slightly higher precision; metric trade-offs remain material.", 988, 486, 210, 90, {
    fontSize: 16,
    color: C.muted,
  });
}

function addAblationSlide(data, index) {
  const slide = presentation.slides.add();
  addChrome(slide, data, index, C.rust);
  addClaim(slide, data, 150, 1070, C.rust);
  addBox(slide, "ablation-input-stem", 160, 336, 145, 4, C.line);
  addBox(slide, "ablation-control-edge", 305, 284, 4, 112, C.line);
  addBox(slide, "ablation-control-out", 305, 284, 110, 4, C.line);
  addBox(slide, "ablation-candidate-out", 305, 392, 110, 4, C.line);
  addBox(slide, "ablation-merge-a", 665, 284, 90, 4, C.line);
  addBox(slide, "ablation-merge-b", 665, 392, 90, 4, C.line);
  addBox(slide, "ablation-merge-v", 755, 284, 4, 112, C.line);
  addBox(slide, "ablation-merge-out", 755, 338, 105, 4, C.line);
  addBox(slide, "ablation-input", 74, 292, 164, 96, C.tealSoft, { geometry: "roundRect", borderRadius: "rounded-xl", lineFill: C.teal, lineWidth: 1 });
  addText(slide, "ablation-input-text", "SAME INPUT\nSAME SEED", 92, 316, 128, 54, { fontSize: 17, bold: true, alignment: "center" });
  addBox(slide, "ablation-control", 414, 246, 250, 78, C.tealSoft, { geometry: "roundRect", borderRadius: "rounded-xl" });
  addText(slide, "ablation-control-text", "CONTROL\nzero fourth channel", 434, 260, 210, 52, { fontSize: 18, bold: true, color: C.teal, alignment: "center" });
  addBox(slide, "ablation-candidate", 414, 354, 250, 78, C.rustSoft, { geometry: "roundRect", borderRadius: "rounded-xl" });
  addText(slide, "ablation-candidate-text", "CANDIDATE\ncontour fourth channel", 434, 368, 210, 52, { fontSize: 18, bold: true, color: C.rust, alignment: "center" });
  addBox(slide, "ablation-compare", 860, 292, 330, 96, C.graphite, { geometry: "roundRect", borderRadius: "rounded-xl" });
  addText(slide, "ablation-compare-text", "PAIRED GROUP BOOTSTRAP\nseeds + views averaged first", 880, 316, 290, 56, { fontSize: 17, bold: true, color: C.ivory, alignment: "center" });
  addMetric(slide, data, 124, 486, 200, "fit groups", "308", C.teal);
  addMetric(slide, data, 446, 486, 220, "holdout groups", "76", C.rust);
  addMetric(slide, data, 810, 486, 220, "paired seeds", "3", C.graphite);
}

async function addNegativeSlide(data, index) {
  const slide = presentation.slides.add();
  addChrome(slide, data, index, C.rust);
  addClaim(slide, data, 150, 1070, C.rust);
  const bytes = await fs.readFile(path.join(assetDir, "loop206-delta.png"));
  slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt: "Loop206 paired deltas with conditional 95% confidence intervals",
    fit: "contain",
    position: { left: 70, top: 274, width: 770, height: 306 },
  });
  addText(slide, "negative-gate", "REJECT", 898, 296, 270, 62, {
    fontFamily: "Georgia",
    fontSize: 52,
    bold: true,
    color: C.rust,
    alignment: "center",
  });
  addText(slide, "negative-dice", "Dice Δ  -0.0313\n95% CI [-0.0491, -0.0156]", 882, 378, 302, 86, {
    fontSize: 20,
    bold: true,
    alignment: "center",
  });
  addText(slide, "negative-bf1", "BF1 Δ  -0.0147\n95% CI [-0.0308, 0.0010]", 882, 482, 302, 86, {
    fontSize: 18,
    color: C.muted,
    alignment: "center",
  });
  addText(slide, "negative-boundary", "Candidate rejected before protected evaluation.", 882, 584, 302, 38, {
    fontSize: 15,
    bold: true,
    color: C.rust,
    alignment: "center",
  });
}

async function addDemoSlide(data, index) {
  const slide = presentation.slides.add();
  addChrome(slide, data, index);
  addClaim(slide, data, 150, 1070);
  const bytes = await fs.readFile(path.join(assetDir, "qualitative-demo.png"));
  slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt: "Fixed-cache lesion segmentation comparison with authorized ground truth",
    fit: "contain",
    position: { left: 68, top: 266, width: 850, height: 340 },
  });
  addText(slide, "demo-parity", "0/76", 974, 282, 196, 70, {
    fontFamily: "Georgia",
    fontSize: 52,
    bold: true,
    color: C.rust,
    alignment: "center",
  });
  addText(slide, "demo-parity-label", "candidate prior parity", 950, 348, 244, 40, {
    fontSize: 16,
    bold: true,
    color: C.muted,
    alignment: "center",
  });
  addText(slide, "demo-rule", "FIXED CACHE\ncontrol + candidate + authorized truth", 950, 422, 244, 74, {
    fontSize: 18,
    bold: true,
    color: C.teal,
    alignment: "center",
  });
  addText(slide, "demo-upload", "ARBITRARY UPLOAD\ncontrol only", 950, 516, 244, 60, {
    fontSize: 18,
    bold: true,
    color: C.rust,
    alignment: "center",
  });
  addText(slide, "demo-clinical", "Non-clinical research demo; not a diagnosis.", 950, 588, 244, 34, {
    fontSize: 14,
    color: C.muted,
    alignment: "center",
  });
}

function addReproSlide(data, index) {
  const slide = presentation.slides.add();
  addChrome(slide, data, index);
  addClaim(slide, data, 150, 1070);
  const columns = [
    {
      x: 74,
      color: C.teal,
      title: "HASH-BOUND",
      text: "Claims\nTables + figures\nCompiled PDF\nModel IDs\nSource reports",
    },
    {
      x: 452,
      color: C.sand,
      title: "RUNTIME-BOUND",
      text: "Checkpoints\nData\nCaches\nSecrets\nStartup verification",
    },
    {
      x: 830,
      color: C.rust,
      title: "NOT YET CLONE-RUNNABLE",
      text: "Loop191/192 configs\nImplementation modules\nPaired predictions\nPhysical laptop evidence",
    },
  ];
  columns.forEach((column, i) => {
    addBox(slide, `repro-rule-${i}`, column.x, 292, 320, 8, column.color);
    addText(slide, `repro-title-${i}`, column.title, column.x, 320, 320, 34, {
      fontSize: 17,
      bold: true,
      color: column.color === C.sand ? C.graphite : column.color,
    });
    addText(slide, `repro-text-${i}`, column.text, column.x, 374, 320, 188, {
      fontFamily: "Georgia",
      fontSize: 22,
      bold: true,
      color: C.graphite,
    });
  });
  addText(slide, "repro-protected", "Protected test remains sealed.", 74, 588, 1076, 34, {
    fontSize: 18,
    bold: true,
    color: C.rust,
    alignment: "center",
  });
}

function addConclusionSlide(data, index) {
  const slide = presentation.slides.add();
  addChrome(slide, data, index, C.rust);
  addClaim(slide, data, 150, 1070, C.rust);
  const statements = [
    ["01", "Repair leakage before optimizing models."],
    ["02", "Use strong baselines without hiding contract differences."],
    ["03", "Publish negative ablations when they reject a mechanism."],
  ];
  statements.forEach(([number, statement], i) => {
    const y = 292 + i * 84;
    addText(slide, `conclusion-number-${i}`, number, 82, y, 72, 48, {
      fontFamily: "Georgia",
      fontSize: 32,
      bold: true,
      color: i === 2 ? C.rust : C.teal,
    });
    addText(slide, `conclusion-statement-${i}`, statement, 180, y + 2, 950, 52, {
      fontFamily: "Georgia",
      fontSize: 25,
      bold: true,
    });
  });
  addBox(slide, "conclusion-next", 78, 550, 1098, 58, C.graphite);
  addText(slide, "conclusion-next-label", "NEXT DEFENSIBLE EXPERIMENT", 98, 562, 282, 30, {
    fontSize: 14,
    bold: true,
    color: C.sand,
  });
  addText(slide, "conclusion-next-text", "Reproduce both systems under one geometry contract before any protected test.", 392, 560, 760, 34, {
    fontSize: 19,
    bold: true,
    color: C.ivory,
    alignment: "right",
  });
}

addTitleSlide(content.slides[0]);
addLeakageSlide(content.slides[1], 2);
addQuestionsSlide(content.slides[2], 3);
addPipelineSlide(content.slides[3], 4);
addDataSlide(content.slides[4], 5);
addModelsSlide(content.slides[5], 6);
addValidationSlide(content.slides[6], 7);
addAblationSlide(content.slides[7], 8);
await addNegativeSlide(content.slides[8], 9);
await addDemoSlide(content.slides[9], 10);
addReproSlide(content.slides[10], 11);
addConclusionSlide(content.slides[11], 12);

await fs.mkdir(path.dirname(OUTPUT), { recursive: true });
await fs.mkdir(QA_DIR, { recursive: true });

for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(QA_DIR, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text());
}

const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(path.join(QA_DIR, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(OUTPUT);
console.log(JSON.stringify({ output: OUTPUT, slides: presentation.slides.items.length, qa: QA_DIR }));
