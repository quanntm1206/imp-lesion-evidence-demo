"use strict";

const state = {
  content: null,
  slides: [],
  currentIndex: 0,
  notesOpen: false,
  focusNode: null,
};

const slidesRoot = document.getElementById("slides");
const deckRoot = document.getElementById("deck");
const previousButton = document.getElementById("previous");
const nextButton = document.getElementById("next");
const pipelineButton = document.getElementById("pipeline");
const notesToggle = document.getElementById("notes-toggle");
const notesClose = document.getElementById("notes-close");
const notesPanel = document.getElementById("notes-panel");
const notesList = document.getElementById("notes-list");
const counter = document.getElementById("counter");
const progressBar = document.getElementById("progress-bar");
const loadError = document.getElementById("load-error");

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function appendBodyList(parent, items, className = "body-list") {
  const list = element("ul", className);
  for (const item of items || []) {
    list.append(element("li", "", item));
  }
  parent.append(list);
  return list;
}

function appendClaimAndBody(parent, slideData) {
  parent.append(element("p", "claim", slideData.claim));
  appendBodyList(parent, slideData.body);
}

function metricTone(value) {
  return String(value).startsWith("-") ? " metric-negative" : "";
}

function createMetricGrid(metrics) {
  const grid = element("div", "metric-grid");
  for (const metric of metrics || []) {
    const item = element("div", `metric${metricTone(metric.value)}`);
    item.append(element("span", "metric-value", metric.value));
    item.append(element("span", "metric-label", metric.label));
    grid.append(item);
  }
  return grid;
}

function createHeader(slideData, index) {
  const header = element("header", "slide-header");
  const titleGroup = element("div", "");
  titleGroup.append(element("p", "kicker", slideData.kicker));
  const title = element("h2", "slide-title", slideData.title);
  title.id = `${slideData.id}-title`;
  title.tabIndex = -1;
  titleGroup.append(title);
  header.append(titleGroup);
  header.append(element("span", "slide-number", String(index + 1).padStart(2, "0")));
  return header;
}

function createFooter(slideData) {
  const footer = element("footer", "slide-footer");
  footer.append(element("span", "evidence-label", slideData.evidence_label));
  const notice = slideData.id === "s10-demo"
    ? state.content.meta.non_clinical_notice
    : "IMP | evidence-bounded research";
  footer.append(element("span", slideData.id === "s10-demo" ? "non-clinical" : "", notice));
  return footer;
}

function pipelineCurrentIds(slideId) {
  const direct = state.content.pipeline
    .filter((node) => node.target === slideId)
    .map((node) => node.id);
  if (slideId === "s09-negative-result") return ["loop206-ablation"];
  return direct;
}

function createBreadcrumb(slideId) {
  const currentIds = new Set(pipelineCurrentIds(slideId));
  const breadcrumb = element("div", "pipeline-breadcrumb");
  breadcrumb.setAttribute("aria-label", "Pipeline position");
  state.content.pipeline.forEach((node, index) => {
    const item = element(
      "span",
      `breadcrumb-node${currentIds.has(node.id) ? " is-current" : ""}`,
      `${String(index + 1).padStart(2, "0")} ${node.label}`,
    );
    breadcrumb.append(item);
  });
  return breadcrumb;
}

function createBackButton() {
  const button = element("button", "back-pipeline", "Back to Pipeline");
  button.type = "button";
  button.addEventListener("click", () => goToId("s04-pipeline"));
  return button;
}

function createTitleStage(slideData) {
  const stage = element("div", "title-stage");
  const lockup = element("div", "title-lockup");
  lockup.append(element("p", "kicker", slideData.kicker));
  const title = element("h1", "", slideData.title);
  title.id = `${slideData.id}-title`;
  title.tabIndex = -1;
  lockup.append(title);
  lockup.append(element("p", "title-subtitle", state.content.meta.subtitle));
  const meta = element("div", "title-meta");
  meta.append(element("span", "", state.content.meta.author));
  meta.append(element("span", "", state.content.meta.date));
  meta.append(element("span", "", `${state.content.meta.duration_minutes} minute defense`));
  lockup.append(meta);
  stage.append(lockup);
  stage.append(element("div", "title-mark"));
  return stage;
}

function createContaminationStage(slideData) {
  const stage = element("div", "contamination-stage");
  const copy = element("div", "");
  appendClaimAndBody(copy, slideData);
  copy.append(createMetricGrid(slideData.metrics));
  const map = element("div", "contamination-map");
  map.setAttribute("aria-label", "Legacy split boundaries crossed by identities");
  map.append(element("span", "", "patient A"));
  map.append(element("span", "", "patient B"));
  map.append(element("span", "", "patient C"));
  stage.append(copy, map);
  return stage;
}

function createQuestionStage(slideData) {
  const stage = element("div", "question-stage");
  const questions = [slideData.body[0], slideData.body[1]];
  questions.forEach((question, index) => {
    const column = element("section", "question-column");
    column.append(element("span", "question-index", `0${index + 1}`));
    column.append(element("p", "", question.replace(/^RQ[0-9]+: */, "")));
    stage.append(column);
  });
  return stage;
}

function createPipelineStage(slideData) {
  const stage = element("div", "pipeline-stage");
  const intro = element("div", "pipeline-intro");
  intro.append(element("p", "claim", slideData.claim));
  intro.append(element("span", "pipeline-hint", "Click a module | Esc returns here"));
  stage.append(intro);

  const nodes = element("div", "pipeline-nodes");
  state.content.pipeline.forEach((node, index) => {
    const button = element("button", "pipeline-node");
    button.type = "button";
    button.dataset.target = node.target;
    button.dataset.node = node.id;
    button.setAttribute("aria-label", `Open ${node.label} detail`);
    button.append(element("span", "node-index", String(index + 1).padStart(2, "0")));
    button.append(element("span", "node-label", node.label));
    button.addEventListener("click", () => {
      state.focusNode = node.id;
      goToId(node.target, { focusNode: node.id });
    });
    nodes.append(button);
  });
  stage.append(nodes);
  return stage;
}

function createDataStage(slideData) {
  const stage = element("div", "split-layout");
  const copy = element("div", "");
  appendClaimAndBody(copy, slideData);
  stage.append(copy, createMetricGrid(slideData.metrics));
  return stage;
}

function createModelsStage(slideData) {
  const stage = element("div", "detail-stage");
  stage.append(createBreadcrumb(slideData.id));
  const comparison = element("div", "system-comparison");
  const systems = [
    {
      title: "IMP - MiT-B3 U-Net",
      text: "384x384 | CLAHE + percentile stretch + median filter | fixed U-Net decoder",
    },
    {
      title: "nnU-Net v2",
      text: "256x256 | raw RGB | self-configuring plans, augmentation, loss, and inference",
    },
  ];
  systems.forEach((system) => {
    const lane = element("section", "system-lane");
    lane.append(element("h3", "", system.title));
    lane.append(element("p", "", system.text));
    comparison.append(lane);
  });
  stage.append(comparison);
  return stage;
}

function createValidationStage(slideData) {
  const stage = element("div", "detail-stage");
  stage.append(createBreadcrumb(slideData.id));
  const layout = element("div", "split-layout");
  const copy = element("div", "");
  appendClaimAndBody(copy, slideData);
  const bars = element("div", "comparison-bars");
  const rows = [
    ["MiT-B3 Dice", 0.8959, false],
    ["nnU-Net Dice", 0.9019, true],
    ["MiT-B3 BF1", 0.4145, false],
    ["nnU-Net BF1", 0.4369, true],
  ];
  rows.forEach(([label, value, dark]) => {
    const row = element("div", "comparison-row");
    row.append(element("span", "comparison-label", label));
    const track = element("div", "bar-track");
    const fill = element("div", `bar-fill${dark ? " is-dark" : ""}`);
    fill.style.width = `${Math.max(12, Number(value) * 100)}%`;
    track.append(fill);
    row.append(track, element("span", "comparison-value", Number(value).toFixed(4)));
    bars.append(row);
  });
  layout.append(copy, bars);
  stage.append(layout);
  return stage;
}

function createAblationStage(slideData) {
  const stage = element("div", "detail-stage");
  stage.append(createBreadcrumb(slideData.id));
  const layout = element("div", "split-layout");
  const copy = element("div", "");
  appendClaimAndBody(copy, slideData);
  const lanes = element("div", "ablation-lanes");
  const specs = [
    ["Zero-channel control", ["R", "G", "B", "0"], false],
    ["Contour-channel candidate", ["R", "G", "B", "locked contour"], true],
  ];
  specs.forEach(([label, channels, candidate]) => {
    const lane = element("section", `ablation-lane${candidate ? " candidate" : ""}`);
    lane.append(element("h3", "", label));
    for (const channel of channels) lane.append(element("span", "channel", channel));
    lanes.append(lane);
  });
  layout.append(copy, lanes);
  stage.append(layout);
  return stage;
}

function createNegativeStage(slideData) {
  const stage = element("div", "detail-stage");
  stage.append(createBreadcrumb(slideData.id));
  const figure = element("div", "figure-stage");
  const image = element("img", "");
  image.src = "assets/loop206-delta.png";
  image.alt = "Loop206 candidate-minus-control robust Dice and boundary F1 deltas with conditional confidence intervals";
  const caption = element("div", "figure-caption");
  appendClaimAndBody(caption, slideData);
  figure.append(image, caption);
  stage.append(figure);
  return stage;
}

function safeDemoUrl() {
  const configured = new URLSearchParams(window.location.search).get("demo")
    || state.content.meta.default_demo_url;
  try {
    const url = new URL(configured);
    if (!new Set(["http:", "https:"]).has(url.protocol)) throw new Error("unsupported protocol");
    return url.href;
  } catch (_error) {
    return state.content.meta.default_demo_url;
  }
}

function createDemoStage(slideData) {
  const stage = element("div", "detail-stage");
  stage.append(createBreadcrumb(slideData.id));
  const figure = element("div", "figure-stage demo-stage");
  const image = element("img", "");
  image.src = "assets/qualitative-demo.png";
  image.alt = "Three authorized train-screen fixed-cache comparisons showing image, two masks, disagreement, and provider-bound ground truth";
  const caption = element("div", "figure-caption");
  caption.append(element("p", "claim", slideData.claim));
  appendBodyList(caption, slideData.body);
  const link = element("a", "demo-action", "Open evidence workbench ↗");
  link.href = safeDemoUrl();
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  caption.append(link);
  figure.append(image, caption);
  stage.append(figure);
  return stage;
}

function createChallengeStage(slideData) {
  const stage = element("div", "challenge-stage");
  const sections = [
    ["Problem", slideData.challenge.problem],
    ["Response", slideData.challenge.response],
    ["Remaining limitation", slideData.challenge.limitation],
  ];
  sections.forEach(([label, text]) => {
    const section = element("section", "challenge-card");
    section.append(element("h3", "challenge-label", label));
    section.append(element("p", "challenge-copy", text));
    stage.append(section);
  });
  return stage;
}

function createReproducibilityStage(slideData) {
  const stage = element("div", "split-layout");
  const copy = element("div", "");
  appendClaimAndBody(copy, slideData);
  const chain = element("div", "hash-chain");
  const steps = [
    ["Source reports", "Registry paths + SHA-256"],
    ["Evidence registry", "Classes + metric contracts"],
    ["Paper and demo", "Shared labels and values"],
    ["Clean CI", "Tables + audit + PDF"],
  ];
  steps.forEach(([title, text]) => {
    const step = element("div", "hash-step");
    step.append(element("strong", "", title), element("span", "", text));
    chain.append(step);
  });
  stage.append(copy, chain);
  return stage;
}

function createConclusionStage(slideData) {
  const stage = element("div", "conclusion-stage");
  stage.append(element("p", "claim", slideData.claim));
  appendBodyList(stage, slideData.body, "body-list takeaway-list");
  return stage;
}

function createGenericStage(slideData) {
  const stage = element("div", "split-layout");
  const copy = element("div", "");
  appendClaimAndBody(copy, slideData);
  stage.append(copy);
  if (slideData.metrics) stage.append(createMetricGrid(slideData.metrics));
  return stage;
}

function createStage(slideData) {
  const makers = {
    title: createTitleStage,
    contamination: createContaminationStage,
    questions: createQuestionStage,
    pipeline: createPipelineStage,
    data: createDataStage,
    models: createModelsStage,
    validation: createValidationStage,
    ablation: createAblationStage,
    "loop206-delta": createNegativeStage,
    "qualitative-demo": createDemoStage,
    "challenge": createChallengeStage,
    reproducibility: createReproducibilityStage,
    conclusion: createConclusionStage,
  };
  return (makers[slideData.visual] || createGenericStage)(slideData);
}

function renderSlides() {
  slidesRoot.replaceChildren();
  state.slides = state.content.slides.map((slideData, index) => {
    const article = element("article", "slide");
    article.id = slideData.id;
    article.dataset.index = String(index);
    article.setAttribute("aria-labelledby", `${slideData.id}-title`);
    article.setAttribute("aria-hidden", "true");

    if (slideData.visual !== "title") article.append(createHeader(slideData, index));
    const content = element("div", "slide-content");
    const stage = createStage(slideData);
    content.append(stage);
    article.append(content, createFooter(slideData));

    if (["s05-data", "s06-models", "s07-validation", "s08-ablation-design", "s09-negative-result", "s10-demo"].includes(slideData.id)) {
      article.append(createBackButton());
    }

    slidesRoot.append(article);
    return article;
  });
}

function indexForId(id) {
  return state.content.slides.findIndex((slide) => slide.id === id);
}

function currentData() {
  return state.content.slides[state.currentIndex];
}

function updateNotes() {
  notesList.replaceChildren();
  for (const note of currentData().notes) notesList.append(element("li", "", note));
}

function setNotesOpen(open) {
  state.notesOpen = open;
  notesPanel.hidden = !open;
  notesToggle.setAttribute("aria-expanded", String(open));
  if (open) updateNotes();
}

function goToId(id, options = {}) {
  const targetIndex = indexForId(id);
  if (targetIndex >= 0) showSlide(targetIndex, options);
}

function showSlide(targetIndex, options = {}) {
  const bounded = Math.max(0, Math.min(targetIndex, state.slides.length - 1));
  const oldIndex = state.currentIndex;
  const direction = bounded >= oldIndex ? 1 : -1;
  state.currentIndex = bounded;

  state.slides.forEach((slide, index) => {
    slide.classList.toggle("is-active", index === bounded);
    slide.classList.toggle("is-before", index < bounded);
    slide.classList.remove("is-focus-entry");
    slide.setAttribute("aria-hidden", String(index !== bounded));
  });

  const active = state.slides[bounded];
  if (options.focusNode) {
    active.classList.add("is-focus-entry");
    const nodeIndex = state.content.pipeline.findIndex((node) => node.id === options.focusNode);
    active.style.setProperty("--focus-origin", `${Math.max(5, nodeIndex * 17 + 8)}% 55%`);
    active.style.setProperty("--focus-offset", `${(nodeIndex - 2.5) * 7}%`);
  }

  previousButton.disabled = bounded === 0;
  nextButton.disabled = bounded === state.slides.length - 1;
  counter.textContent = `${String(bounded + 1).padStart(2, "0")} / ${String(state.slides.length).padStart(2, "0")}`;
  progressBar.style.width = `${((bounded + 1) / state.slides.length) * 100}%`;
  updateNotes();

  const id = currentData().id;
  if (window.location.hash !== `#${id}`) history.pushState(null, "", `#${id}`);
  const title = active.querySelector("h1, h2");
  window.setTimeout(() => title?.focus({ preventScroll: true }), 30);
  document.title = `${currentData().title} | Evidence Before Leaderboards`;
  window.dispatchEvent(new CustomEvent("deck:slidechange", { detail: { id, index: bounded, direction } }));
}

function handleKey(event) {
  if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.target instanceof HTMLAnchorElement || event.target instanceof HTMLButtonElement) {
    if (event.key === "Enter" || event.key === " ") return;
  }
  if (event.key === "ArrowRight" || event.key === "PageDown") {
    event.preventDefault();
    showSlide(state.currentIndex + 1);
  } else if (event.key === "ArrowLeft" || event.key === "PageUp") {
    event.preventDefault();
    showSlide(state.currentIndex - 1);
  } else if (event.key === "Home") {
    event.preventDefault();
    showSlide(0);
  } else if (event.key === "End") {
    event.preventDefault();
    showSlide(state.slides.length - 1);
  } else if (event.key === "Escape") {
    event.preventDefault();
    if (state.notesOpen) setNotesOpen(false);
    else goToId("s04-pipeline");
  } else if (event.key.toLowerCase() === "n") {
    event.preventDefault();
    setNotesOpen(!state.notesOpen);
  } else if (event.key.toLowerCase() === "f") {
    event.preventDefault();
    if (!document.fullscreenElement) deckRoot.requestFullscreen?.();
    else document.exitFullscreen?.();
  }
}

function bindControls() {
  previousButton.addEventListener("click", () => showSlide(state.currentIndex - 1));
  nextButton.addEventListener("click", () => showSlide(state.currentIndex + 1));
  pipelineButton.addEventListener("click", () => goToId("s04-pipeline"));
  notesToggle.addEventListener("click", () => setNotesOpen(!state.notesOpen));
  notesClose.addEventListener("click", () => setNotesOpen(false));
  document.addEventListener("keydown", handleKey);
  window.addEventListener("hashchange", () => {
    const target = window.location.hash.slice(1);
    const index = indexForId(target);
    if (index >= 0 && index !== state.currentIndex) showSlide(index);
  });
}

async function initialize() {
  try {
    const response = await fetch("content.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`content request failed: ${response.status}`);
    state.content = await response.json();
    renderSlides();
    bindControls();
    const hashIndex = indexForId(window.location.hash.slice(1));
    state.currentIndex = hashIndex >= 0 ? hashIndex : 0;
    showSlide(state.currentIndex);
    deckRoot.setAttribute("aria-busy", "false");
    window.__deck = {
      goToId,
      currentId: () => currentData().id,
      slideCount: () => state.slides.length,
      pipelineTargets: () => state.content.pipeline.map((node) => node.target),
    };
  } catch (error) {
    console.error("presentation initialization failed", error);
    deckRoot.hidden = true;
    loadError.hidden = false;
  }
}

initialize();
