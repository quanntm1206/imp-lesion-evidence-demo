# Interactive Research Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify an English 12-slide interactive research-defense deck with HTML, PPTX, and PDF deliverables.

**Architecture:** A dependency-free HTML presentation owns nonlinear navigation and motion. A plain JavaScript `@oai/artifact-tool` builder generates the editable PPTX from the same content contract. Tracked paper figures provide evidence visuals; deterministic scripts create raster derivatives and static exports.

**Tech Stack:** HTML5, CSS, vanilla JavaScript, Python stdlib tests, `@oai/artifact-tool`, Poppler, LibreOffice or PowerPoint-compatible PDF export, Codex browser QA.

## Global Constraints

- English, 16:9, 12 audience-facing slides, 10-12 minute talk.
- Use only paper/registry/manifest-bound claims and values.
- Keep protected-test and clinical claims explicitly out of scope.
- Preserve Evidence Atlas tokens and accessible reduced-motion behavior.
- Keep scratch artifacts outside Git; track only source and final deliverables.
- HTML is the interactive primary; PPTX/PDF are linear fallbacks.

---

### Task 1: Content Contract And Tests

**Files:**
- Create: `presentation/interactive/content.json`
- Create: `tests/presentation/test_interactive_deck.py`

**Interfaces:**
- Consumes: tracked paper, registry, artifact manifest.
- Produces: `content.json` with `meta`, `pipeline`, and exactly 12 `slides`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_content_contract_has_exact_slide_and_pipeline_targets():
    content = json.loads(CONTENT.read_text(encoding="utf-8"))
    assert len(content["slides"]) == 12
    slide_ids = {slide["id"] for slide in content["slides"]}
    assert {node["target"] for node in content["pipeline"]} <= slide_ids

def test_scientific_claims_are_bounded():
    text = CONTENT.read_text(encoding="utf-8")
    for value in ["0.8959", "0.9019", "-0.0313", "-0.0491", "-0.0156"]:
        assert value in text
    assert "protected test remains sealed" in text.lower()
    assert "state of the art" in text.lower()
    assert "not" in text.lower()
```

- [ ] **Step 2: Run RED**

Run: `E:\0. IMP\.venv-win\Scripts\python.exe -m pytest tests/presentation/test_interactive_deck.py -q`

Expected: FAIL because `content.json` does not exist.

- [ ] **Step 3: Add the 12-slide content contract**

Create `content.json` with exact slide IDs `s01-title` through
`s12-conclusion`, pipeline targets `s05-data`, `s06-models`, `s07-validation`,
`s08-ablation-design`, `s09-negative-result`, and `s10-demo`, evidence labels,
speaker notes, and source paths.

- [ ] **Step 4: Run GREEN**

Run the Task 1 command. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add presentation/interactive/content.json tests/presentation/test_interactive_deck.py
git commit -m "test(deck): lock scientific content contract"
```

---

### Task 2: Bound Visual Assets

**Files:**
- Create: `presentation/interactive/assets/loop206-delta.png`
- Create: `presentation/interactive/assets/qualitative-demo.png`
- Create: `presentation/interactive/assets/asset-manifest.json`
- Create: `scripts/presentation/build_deck_assets.ps1`
- Modify: `tests/presentation/test_interactive_deck.py`

**Interfaces:**
- Consumes: tracked `loop206_delta.pdf`, `qualitative_demo.pdf`, and artifact manifest.
- Produces: deterministic PNG derivatives plus source/output SHA-256 bindings.

- [ ] **Step 1: Add failing asset-binding tests**

```python
def test_asset_manifest_binds_source_and_output_bytes():
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["assets"]:
        assert sha256(ROOT / entry["source"]) == entry["source_sha256"]
        assert sha256(ROOT / entry["output"]) == entry["output_sha256"]
```

- [ ] **Step 2: Run RED**

Expected: FAIL because the derivative manifest is absent.

- [ ] **Step 3: Implement deterministic rasterization**

Use installed Poppler `pdftoppm` with fixed PNG resolution. Hash source and
output bytes with `Get-FileHash`. Write project-relative paths only.

- [ ] **Step 4: Run asset builder and tests**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/presentation/build_deck_assets.ps1
E:\0. IMP\.venv-win\Scripts\python.exe -m pytest tests/presentation/test_interactive_deck.py -q
```

Expected: PASS; no absolute path in manifest.

- [ ] **Step 5: Commit**

```powershell
git add scripts/presentation presentation/interactive/assets tests/presentation/test_interactive_deck.py
git commit -m "build(deck): bind presentation assets"
```

---

### Task 3: Interactive HTML Deck

**Files:**
- Create: `presentation/interactive/index.html`
- Create: `presentation/interactive/deck.css`
- Create: `presentation/interactive/deck.js`
- Modify: `tests/presentation/test_interactive_deck.py`

**Interfaces:**
- Consumes: `content.json` and bound assets.
- Produces: hash-addressable slide navigation and presenter controls.

- [ ] **Step 1: Add failing HTML contract tests**

```python
def test_html_exposes_accessible_navigation_and_reduced_motion():
    html = INDEX.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    assert 'aria-live="polite"' in html
    assert "prefers-reduced-motion" in css
    assert "hashchange" in js
    assert "Escape" in js
    assert "Back to Pipeline" in js
```

- [ ] **Step 2: Run RED**

Expected: FAIL because HTML/CSS/JS files are absent.

- [ ] **Step 3: Implement semantic shell and Evidence Atlas tokens**

Create a full-screen 16:9 stage, slide list, progress rail, evidence footer,
presenter-notes overlay, and responsive mobile composition. Use Georgia and
Trebuchet MS, ivory/graphite/teal/rust tokens, and no external dependencies.

- [ ] **Step 4: Implement navigation and motion**

Load `content.json`, render 12 slides, wire pipeline module targets and Back
controls, update hash/history, support click/touch/keyboard, move focus, and
apply reduced-motion fallback. Resolve demo URL from `?demo=` with loopback as
the default and never persist it.

- [ ] **Step 5: Run GREEN**

Run Task 1 tests. Expected: PASS.

- [ ] **Step 6: Browser functional check**

Serve the worktree locally. Verify six hub targets, six Back paths, deep link,
arrow/Home/End/Escape navigation, notes toggle, and zero console errors.

- [ ] **Step 7: Commit**

```powershell
git add presentation/interactive tests/presentation/test_interactive_deck.py
git commit -m "feat(deck): add interactive evidence presentation"
```

---

### Task 4: Editable PPTX Fallback

**Files:**
- Create: `scripts/presentation/build_pptx.mjs`
- Create: `outputs/imp-lesion-evidence-defense.pptx`
- Modify: `tests/presentation/test_interactive_deck.py`

**Interfaces:**
- Consumes: shared content JSON and PNG assets.
- Produces: editable 12-slide 1280x720 PPTX with speaker notes.

- [ ] **Step 1: Add failing output test**

```python
def test_final_pptx_exists_and_is_nontrivial():
    assert PPTX.exists()
    assert PPTX.stat().st_size > 100_000
```

- [ ] **Step 2: Initialize artifact-tool scratch workspace**

Run `setup_artifact_tool_workspace.mjs` with an external temp workspace. Keep
the generated `.mjs` source tracked in `scripts/presentation`; keep tool caches
outside the repository.

- [ ] **Step 3: Build the deck with artifact-tool**

Use `Presentation.create({ slideSize: { width: 1280, height: 720 } })`. Create
editable text, simple pipeline shapes, two evidence charts/images, evidence
footers, and speaker notes. Export slide PNG previews, layout JSON, a montage,
and the final PPTX.

- [ ] **Step 4: Validate structure and overflow**

```powershell
python "$SKILL_DIR/container_tools/render_slides.py" outputs/imp-lesion-evidence-defense.pptx
python "$SKILL_DIR/container_tools/slides_test.py" outputs/imp-lesion-evidence-defense.pptx
```

Expected: 12 rendered slides; zero overflow; zero unintended overlap.

- [ ] **Step 5: Commit**

```powershell
git add scripts/presentation/build_pptx.mjs outputs/imp-lesion-evidence-defense.pptx tests/presentation/test_interactive_deck.py
git commit -m "feat(deck): add editable PowerPoint fallback"
```

---

### Task 5: Portable HTML And PDF Fallbacks

**Files:**
- Create: `scripts/presentation/package_deck.ps1`
- Create: `outputs/imp-lesion-evidence-defense.html`
- Create: `outputs/imp-lesion-evidence-defense.pdf`
- Create: `outputs/imp-lesion-evidence-defense-manifest.json`
- Modify: `tests/presentation/test_interactive_deck.py`

**Interfaces:**
- Consumes: verified HTML source and PPTX.
- Produces: self-contained HTML, 12-page PDF, and SHA-256 manifest.

- [ ] **Step 1: Add failing package tests**

```python
def test_delivery_manifest_binds_outputs():
    receipt = json.loads(DELIVERY_MANIFEST.read_text(encoding="utf-8"))
    assert receipt["slide_count"] == 12
    for entry in receipt["files"]:
        assert sha256(ROOT / entry["path"]) == entry["sha256"]
```

- [ ] **Step 2: Run RED**

Expected: FAIL because package outputs do not exist.

- [ ] **Step 3: Implement packaging**

Inline CSS, JavaScript, JSON, and PNG assets into one HTML file. Export PPTX
to PDF with the available Office-compatible tool. Use `pdfinfo` to require 12
pages. Write relative paths, byte sizes, and hashes to the manifest.

- [ ] **Step 4: Run GREEN**

Run presentation tests. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/presentation/package_deck.ps1 outputs tests/presentation/test_interactive_deck.py
git commit -m "build(deck): package portable presentation"
```

---

### Task 6: Visual QA, Research Audit, And Handoff

**Files:**
- Modify only files with verified visual or content defects.
- Create scratch QA outputs outside Git.

**Interfaces:**
- Consumes: all final deliverables.
- Produces: verified release state and GitHub-synced branch.

- [ ] **Step 1: Run full automated gates**

```powershell
E:\0. IMP\.venv-win\Scripts\python.exe -m pytest tests/presentation tests/demo -q
git diff --check
```

- [ ] **Step 2: Perform HTML design review**

Capture 1440x900 and 390x844 screenshots. Inspect default, hub, each module,
notes overlay, and live-demo callout. Check hierarchy, contrast, focus, motion,
reduced motion, responsive layout, and console errors.

- [ ] **Step 3: Perform PPTX/PDF visual review**

Inspect all 12 PPTX slide PNGs individually, then the montage. Render all PDF
pages and confirm typography, charts, tables, captions, footers, and page count.

- [ ] **Step 4: Audit scientific language**

Search all audience-facing copy for `SOTA`, `superior`, `protected test`,
`clinical`, `diagnostic`, and exact metric values. Confirm every occurrence is
bounded by the paper evidence contract.

- [ ] **Step 5: Fix defects and rerun gates**

Repeat Tasks 6.1-6.4 until no Critical, Important, overflow, clipping, broken
link, console error, or unsupported claim remains.

- [ ] **Step 6: Commit and push**

```powershell
git add presentation scripts/presentation outputs tests/presentation
git commit -m "fix(deck): complete visual and evidence QA"
git push -u origin feature/interactive-research-deck
```

