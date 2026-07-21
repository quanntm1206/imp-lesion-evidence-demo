# Interactive Research Deck Design

## Goal

Create a complete English research-defense deck for a 10-12 minute talk. The
primary deliverable is an offline-capable interactive HTML presentation. A
verified PowerPoint deck and PDF provide linear fallbacks.

## Communication Job

By the end, an academic audience should understand why evidence discipline
matters more than a favorable leaderboard number, because the repaired study
shows an honest baseline comparison and a scientifically useful negative
ablation without opening protected test data.

## Evidence Contract

- Use only claims recorded in `paper/clean_v3_loop206`, the evidence registry,
  and the bound artifact manifest.
- Label Loop191/192 as adaptive development-validation, single-run,
  selection-optimistic point estimates.
- Label Loop206 as a three-seed train-screen ablation with uncertainty
  conditional on the selected seeds.
- State that protected test-v3 and PH2 remain sealed.
- State that the contour-channel candidate failed the preregistered gate.
- Do not claim statistical superiority, protected-test performance, clinical
  utility, diagnosis, or state of the art.
- Preserve exact values: 2,869 total images; 2,008 train; 431 validation; 430
  sealed test; robust Dice 0.8959 versus 0.9019; BF1 0.4145 versus 0.4369;
  Loop206 robust-Dice delta -0.0313 with CI [-0.0491, -0.0156]; BF1 delta
  -0.0147 with CI [-0.0308, 0.0010].

## Narrative

The deck follows context -> evidence repair -> controlled comparison ->
negative result -> demonstration -> implications.

1. Evidence before leaderboard claims.
2. Leakage turned a benchmark into historical evidence.
3. Two bounded research questions.
4. Interactive evidence pipeline hub.
5. Clean-v3 repairs the comparison contract.
6. The models differ as complete systems.
7. nnU-Net records higher development-validation point estimates.
8. Loop206 isolates one contour-channel change.
9. The contour channel fails its primary gate.
10. The demo exposes only evidence the registry authorizes.
11. Reproducibility is strong for claims, incomplete for full retraining.
12. The defensible conclusion is narrower and more useful.

## Interactive Model

Slide 4 is the hub. It contains six focusable pipeline modules:

`Data Audit -> Preprocessing -> Models -> Robust Evaluation -> Loop206 Ablation -> Evidence-bound Demo`

Each module navigates to its detail slide. Each detail slide includes a compact
pipeline breadcrumb and a `Back to Pipeline` control. Navigation supports
mouse, touch, Enter, Space, arrows, Home, End, and Escape. The URL hash tracks
the active slide for deep links. Focus moves to the new slide title.

Transitions last 520 ms with a cubic-bezier easing. The hub module expands
toward the detail composition while the remaining pipeline fades. A
`prefers-reduced-motion` branch removes movement and keeps a short opacity
change.

## Visual Language

Name: `Evidence Atlas`.

- Canvas: warm ivory `#F4EFE4`.
- Ink: graphite `#1D211F`.
- Validated evidence: teal `#177D76`.
- Contamination and rejected hypotheses: rust `#B54E36`.
- Quiet structural fields: sage `#DDE7DF` and sand `#D8CFBF`.
- Headline type: Georgia, editorial and research-oriented.
- Body type: Trebuchet MS, clear at projection distance.
- Minimum sizes: 50 pt deck title, 35 pt slide title, 24 pt callouts, 16 pt
  body in PPTX equivalents.
- Shapes use square or lightly softened corners, thin rules, no generic card
  grid, no decorative gradients, and no stock medical imagery.
- Motion explains hierarchy; it is not decoration.

## Visual Assets

- Reuse the bound `loop206_delta.pdf` as the main quantitative chart.
- Reuse the bound `qualitative_demo.pdf` as the demonstration evidence.
- Use simple native shapes for the interactive pipeline and model comparison.
- Raster derivatives are generated from tracked PDFs; no raw dataset images,
  checkpoints, caches, or absolute provenance paths enter the deck.

## Deliverables

- `presentation/interactive/index.html`: interactive deck entry point.
- `presentation/interactive/deck.css`: tokens, layouts, transitions, responsive
  behavior, print behavior.
- `presentation/interactive/deck.js`: navigation, deep links, presenter notes,
  demo URL override, accessibility behavior.
- `presentation/interactive/assets/`: bound raster derivatives used by HTML.
- `outputs/imp-lesion-evidence-defense.pptx`: editable linear fallback.
- `outputs/imp-lesion-evidence-defense.pdf`: static fallback.
- `outputs/imp-lesion-evidence-defense.html`: portable HTML entry copy.

## Validation

- Unit/contract tests validate slide count, links, evidence values, labels,
  disclaimer language, and reduced-motion support.
- Browser tests exercise every hub module, Back control, keyboard navigation,
  hash deep links, desktop layout, and mobile layout.
- PPTX QA renders all slides, runs overflow checks, and visually inspects every
  slide at full size.
- HTML QA captures desktop and mobile screenshots and checks console errors.
- PDF QA checks page count and renders a montage.
- All outputs must remain usable without network access, except an explicitly
  configured live-demo URL.

## Known Limits

- The HTML deck provides the full interactive experience. PPTX and PDF remain
  linear fallbacks; no claim is made that PDF preserves transitions.
- The live demo URL is runtime-configurable. The deck defaults to
  `http://127.0.0.1:7860`; no public URL or tunnel token is stored.
- Physical RTX 4060 execution remains unverified. This machine can rehearse the
  workflow on its RTX 5060 Ti without relabeling that rehearsal as laptop
  evidence.
