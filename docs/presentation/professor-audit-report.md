# HISTORICAL / SUPERSEDED - Professor A Audit Report

> **Historical snapshot only.** This report must not be treated as current release evidence. Its 13-page, 12-slide/HTML, old Slide 10, Slides 11-15 route, and `01 / 12` counter observations describe the inspected pre-remediation artifacts below.
>
> **Authoritative current state.** Read status only from `paper/clean_v3_loop206/artifact_manifest.json`, `release/imp_release_manifest.json`, and `outputs/imp-lesion-evidence-defense-manifest.json`. The paper is `status=current`; the presentation is `package_state=complete` with 17 slides; both bind release digest `435606d5adc296be57405c65a9c725af3dff96c15f9aabf7ac0924d06387a264`. Live runtime, browser, and tunnel acceptance remain `unverified/blocked` as disclosed by `docs/presentation/presenter-s-transcript.md`. Any use of "current" in the historical body below refers only to the inspection snapshot, not today's release.

Date: 2026-07-23

## Scope and question

Question: Is the current paper, defense package, live-demo evidence, and P1
scientific plan ready for an authoritative oral defense and release?

Scope: read-only inspection of the paper source/PDF state, interactive source,
HTML/PPTX/PDF package and manifests, PPTX build/navigation evidence, defense
question bank, Presenter S transcript, Task 4 runtime report/final review, Task
5 P1 preflight/report/review, and Task 6 readiness/report/review. This audit did
not run training, Docker, Gradio, a browser, Cloudflare, or Git mutation. It did
not open test-v3, PH2, private images, masks, model tensors, or checkpoints.

Severity:

- **P1:** blocks authoritative defense/release or the planned scientific gate.
- **P2:** material research weakness; bounded wording prevents an immediate
  overclaim, but stronger interpretation is not admissible.
- **P3:** clarity, maintenance, or rehearsal defect with a contained effect.

## Evidence classification

| Class | Audit statement | Exact evidence |
| --- | --- | --- |
| **Observed** | Current TeX source contains the bounded historical comparison, negative Loop206 ablation, live-lane limitations, and blocked P1 wording. | `paper/clean_v3_loop206/main.tex`; `paper/clean_v3_loop206/sections/03_data_protocol.tex`; `paper/clean_v3_loop206/sections/05_experiments.tex`; `paper/clean_v3_loop206/sections/06_results.tex`; `paper/clean_v3_loop206/sections/08_limitations_ethics.tex`; `paper/clean_v3_loop206/sections/09_reproducibility.tex` |
| **Observed** | Paper PDF is 13 pages and presentation PDF is 12 pages. Their manifests classify them as stale. | `paper/clean_v3_loop206/main.pdf`; `paper/clean_v3_loop206/artifact_manifest.json`; `outputs/imp-lesion-evidence-defense.pdf`; `outputs/imp-lesion-evidence-defense-manifest.json` |
| **Observed** | `content.json` has 17 slides. The current PPTX file has 17 slides, six pipeline jumps, six Back jumps, and a medium fade on all 17 slides. The portable HTML embeds an older 12-slide content object. | `presentation/interactive/content.json`; `outputs/imp-lesion-evidence-defense.pptx`; `outputs/imp-lesion-evidence-defense.html`; `.superpowers/sdd/task-6-navigation-report.md`; `scripts/presentation/inject_pptx_navigation.py` |
| **Observed** | Presenter S completed launcher/callback steps but did not visually render or interact with the UI in a browser. S recorded cleanup exit 5. | `docs/presentation/presenter-s-transcript.md` |
| **Observed** | Current reconstructed nnU-Net repeat attempts produced unequal same-input mask bytes. Task 4 final review therefore keeps acceptance blocked. | `.superpowers/sdd/task-4-runtime-browser-resume-report.md`; `.superpowers/sdd/task-4-runtime-browser-final-review.md` |
| **Observed** | P1 protocol scaffolding exists, but the data report, model/runtime/input manifests, six configs, and six job receipts do not. | `experiments/rq1_v2/protocol.json`; `.superpowers/sdd/completion-task5-p1-preflight-report.md`; `.superpowers/sdd/completion-task5-p1-preflight-review.md` |
| **Established** | A hash binds bytes or an identity record. It does not, by itself, establish a scientific or deployment outcome. | `demo/data/evidence_registry.json`; `release/imp_release_manifest.json`; `docs/presentation/defense-question-bank.md` |
| **Established** | The recorded RQ1 comparison is adaptive development-validation, single-run, and geometry-limited. The Loop206 interval is conditional on three selected seeds. | `paper/clean_v3_loop206/main.tex`; `paper/clean_v3_loop206/sections/03_data_protocol.tex`; `paper/clean_v3_loop206/sections/05_experiments.tex`; `paper/clean_v3_loop206/sections/06_results.tex`; `demo/data/evidence_registry.json` |
| **Assumption** | Task reports describe distinct attempts and run IDs. An earlier exit 5 and a later exit 0 are not interchangeable unless one canonical acceptance packet binds the exact run. | `.superpowers/sdd/task-4-runtime-browser-resume-report.md`; `docs/presentation/presenter-s-transcript.md` |
| **Assumption** | Static OOXML/text inspection establishes structure, not rendered visual quality. | `outputs/imp-lesion-evidence-defense.pptx`; `docs/presentation/presenter-s-transcript.md` |
| **Speculation** | Explanations for nnU-Net drift or the contour-channel failure remain hypotheses until isolated experiments support them. | `.superpowers/sdd/task-4-determinism-analysis.md`; `paper/clean_v3_loop206/sections/07_discussion.tex` |

## Findings

### P1-01 - The authoritative paper/deck package is split across stale and current artifacts

**Observed.** The paper manifest records `main.pdf` as `stale_uncompiled`, built
against release digest `c09a...`, while the current release digest is
`d33d...`. The PDF is 13 pages; Task 6 records an isolated current-source build
of 14 pages that was not promoted. The presentation manifest records
`slide_count=12`, `package_state=incomplete_blocked`, a stale 12-page PDF, and
a stale/rebuild-blocked PPTX entry. The actual PPTX now contains 17 slides, but
its current bytes/hash are not the bytes/hash recorded by that manifest.

The HTML is more serious: its manifest entry says `current`, yet decoding the
embedded JSON yields only 12 slides (`s01` through old `s12`). It retains older
Slide 10 wording that presents hash receipts without stating that no canonical
live receipt is included, and older reproducibility wording that calls claims
"hash-verifiable." Current `content.json` has 17 slides and corrected bounded
wording. The local `index.html` fallback counter still says `01 / 12`.

The packaging script is not aligned with the 17-slide source: it validates
slides 1-12, requires a 12-page PDF, and writes `slide_count=12`.

**Risk.** A committee can open a file with old claims or incomplete challenge
slides while source reviewers inspect a different 17-slide deck. Filename and
release digest do not resolve the contradiction.

**Evidence.** `paper/clean_v3_loop206/artifact_manifest.json`;
`paper/clean_v3_loop206/main.pdf`;
`.superpowers/sdd/task-6-paper-evidence-update-report.md`;
`.superpowers/sdd/task-6-paper-evidence-review.md`;
`presentation/interactive/content.json`;
`presentation/interactive/index.html`;
`outputs/imp-lesion-evidence-defense.html`;
`outputs/imp-lesion-evidence-defense.pptx`;
`outputs/imp-lesion-evidence-defense.pdf`;
`outputs/imp-lesion-evidence-defense-manifest.json`;
`scripts/presentation/package_deck.ps1`.

**Acceptance criterion.** One clean package operation must produce: (1) a
current paper PDF from the current TeX source; (2) 17-slide HTML, PPTX, and PDF
from the same `content.json`; (3) exact file bytes/sizes/SHA-256 values in both
artifact manifests; (4) the current release digest in every projection; (5)
`package_state=complete` and `slide_count=17`; (6) the corrected receipt and
source-byte wording in every format; and (7) full visual review of every
current PDF page and PPTX slide. The stale files cannot be used as visual QA for
the current source.

### P1-02 - P1 scientific rerun has no admissible six-job evidence

**Observed.** `experiments/rq1_v2/protocol.json` lists seeds 206, 1206, and
2206, but `dataset_index_status` is `unresolved_blocked` and the dataset-index
digest is null. Current inspection found no
`data_integrity_report.json`, `model_artifacts.json`, arm runtime manifests,
`experiment_input_manifest.json`, `configs/rq1_v2/`, job-receipt directory, or
runtime job directory. A seed list is not a completed run.

The exact missing scientific contract remains: two arms times three seeds, six
locked configs, six immutable job receipts, three execution pairs, all nine
crossed seed contrasts, independent-arm/group bootstrap, and preregistered
error slices.

**Risk.** No new P1 metric, interval, uncertainty statement, error-slice result,
or reproducibility result can be admitted. Historical/fixed/live evidence
cannot substitute for the missing prospective jobs.

**Evidence.** `experiments/rq1_v2/protocol.json`;
`.superpowers/sdd/completion-task5-p1-preflight-report.md`;
`.superpowers/sdd/completion-task5-p1-preflight-review.md`;
`.superpowers/sdd/tasks8-13-readiness-audit.md`;
`.superpowers/sdd/task11-compute-feasibility-review.md`;
`paper/clean_v3_loop206/sections/09_reproducibility.tex`;
`reports/paper_revision/manuscript_readiness_audit.md`.

**Acceptance criterion.** Supply and hash-bind the authorized Clean-v3 index;
pass the zero-crossing data-integrity audit; bind the real MiT-B3 tensor state,
both arm runtimes/locks/images, shared condition/metric contracts, and six
seed-only-different configs; pass exact workstation/laptop preflights without
changing the locked protocol; complete and validate all six job receipts; then
generate nine crossed contrasts, independent-arm/group bootstrap, required
slices, and a digest-bound RQ1-v2 report. Only that validated report may enter
paper/deck projections.

### P1-03 - Reconstructed nnU-Net repeatability remains failed; one public sample exceeds the transport contract

**Observed.** Task 4 records matching same-input/model/checkpoint/geometry
bindings but unequal repeated masks: an earlier attempt differed at 83 pixels;
the final rebuilt configuration differed at 87 pixels. Both attempts fail the
same-current-runtime A/B/A gate. This is observed current reconstructed-runtime
nondeterminism, not an inferred request/PNG binding defect. The final review explicitly says `SPEC: FAIL`
and keeps Task 4 blocked.

`ISIC_0016069` decodes to `2848x4288x3`. Lossless PNG-in-JSON exceeds the
pinned 16 MiB request cap. Local encoding rejects before HTTP; a bounded
sidecar probe returns 413. This is a transport admission boundary, not evidence
about the model output.

The source-level reconstructed-runtime disclosure is adequate: the paper,
current slide source, release manifest, and question bank state reconstructed
runtime and deny original-runtime equivalence. That disclosure does not cure
failed repeatability.

**Risk.** A live receipt cannot be promoted as deterministic. The oversized
sample cannot be described as a successful raw API inference. A visible mask
must not be treated as Paper RQ1 evidence.

**Evidence.** `.superpowers/sdd/task-4-runtime-browser-resume-report.md`;
`.superpowers/sdd/task-4-runtime-browser-final-review.md`;
`.superpowers/sdd/task-4-determinism-analysis.md`;
`paper/clean_v3_loop206/sections/06_results.tex`;
`paper/clean_v3_loop206/sections/08_limitations_ethics.tex`;
`paper/clean_v3_loop206/sections/09_reproducibility.tex`;
`presentation/interactive/content.json`;
`release/imp_release_manifest.json`;
`docs/presentation/defense-question-bank.md`.

**Acceptance criterion.** One exact pinned reconstructed image must pass a
persistent A/B/A probe: same canonical A bytes and bindings produce identical
decoded mask bytes/hash; B has different input and output hashes; immutable
path-free evidence records runtime controls. Then rerun the complete dual path.
For `ISIC_0016069`, either admit it under one deliberately revised and atomically
pinned size contract, or keep it as an explicit fail-closed 413 example. Do not
silently resize, bypass, or call the current pre-HTTP rejection a completed raw
API diagnostic.

### P1-04 - Presenter S has no browser/UI acceptance evidence

**Observed.** S recorded 17-slide source structure, callback success/failure/
recovery, and a local HTTP 200. S explicitly records browser discovery failure
and classifies desktop `1440x900`, mobile `390x844`, pipeline navigation,
challenge interaction, focus, responsive layout, and screenshots as
unverified. This is the correct status for S's rehearsal.

Task 4 later records transient controller-owned desktop/mobile checks and an
ephemeral public GET. Those observations are operational evidence for that
attempt, not S browser evidence; no screenshot or browser receipt was retained
in the package. Static PPTX navigation inspection shows links exist, not that
the rendered HTML interactions or challenge layouts work.

**Risk.** A presenter can mistakenly convert source structure or another
controller's transient observation into a personal/browser acceptance claim.

**Evidence.** `docs/presentation/presenter-s-transcript.md`;
`.superpowers/sdd/task-4-runtime-browser-resume-report.md`;
`.superpowers/sdd/task-4-runtime-browser-final-review.md`;
`.superpowers/sdd/task-7-report.md`;
`.superpowers/sdd/task-6-navigation-report.md`;
`outputs/imp-lesion-evidence-defense.pptx`.

**Acceptance criterion.** S, or a named verifier whose evidence is incorporated
into S's transcript, must render current 17-slide HTML and current Gradio UI at
both required viewports; exercise default, loading, dual success, oversize
failure, stale-output clearing, and recovery; test all six pipeline jumps,
Back/Escape behavior, five challenge cards, focus, reduced motion, contrast,
and horizontal overflow; retain sanitized screenshots plus an immutable
acceptance receipt bound to the current release/source hashes. Until then S
must continue saying UI/browser state is unverified.

### P1-05 - Shutdown evidence is inconsistent across attempts and has no single canonical closure

**Observed.** S's rehearsal ended with exit 5 although ports closed. Task 4's
earlier final clean run also recorded exit 5 after Gradio descendants exceeded
the wait. A later Task 4 sidecar-only/clean shutdown entry records exit 0 and
states that it supersedes the earlier failure for that run. The paper still
describes the recorded ordered shutdown as exit 5. These are not necessarily
false reports; they are different attempts with different acceptance scope.

**Risk.** Selecting the favorable exit 0 without binding the complete browser/
tunnel/dual run creates an unsupported cleanup-closure statement. Closed ports
alone do not establish ordered owned-process cleanup.

**Evidence.** `docs/presentation/presenter-s-transcript.md`;
`.superpowers/sdd/task-4-runtime-browser-resume-report.md`;
`.superpowers/sdd/task-4-runtime-browser-final-review.md`;
`paper/clean_v3_loop206/sections/08_limitations_ethics.tex`.

**Acceptance criterion.** A single named run ID must bind successful sidecar,
Gradio, browser, tunnel, and ordered stop events. Final stop must exit 0, prove
the owned Cloudflare/Gradio/sidecar identities stopped in order, prove ports
7860/7861/7862 closed, and retain append-only stop records. Paper, S transcript,
runbook, and manifest must all point to that same canonical status while older
failures remain historical.

### P2-01 - RQ1 is a strong system baseline comparison, not a fair component-level causal comparison

**Observed.** nnU-Net is a credible strong system-level comparator. However,
the two systems differ in preprocessing, resolution, decoder, loss,
augmentation, schedule, postprocessing, and selection policy. Loop192 predicts
at 256x256 and is resized to a 384x384 metric canvas, while the IMP control
operates at 384x384. The release does not include a complete simple baseline or
matched component ablation for RQ1.

The manuscript states these limitations correctly. Therefore the current
descriptive direction is admissible; encoder, architecture, or mechanism
attribution is not.

**Evidence.** `paper/clean_v3_loop206/sections/03_data_protocol.tex`;
`paper/clean_v3_loop206/sections/04_methods.tex`;
`paper/clean_v3_loop206/sections/05_experiments.tex`;
`paper/clean_v3_loop206/sections/07_discussion.tex`;
`paper/clean_v3_loop206/sections/08_limitations_ethics.tex`;
`reports/paper_revision/manuscript_readiness_audit.md`.

**Acceptance criterion.** For a stronger RQ1 interpretation, freeze one
original-image geometry contract, identical identities/conditions/metric code,
comparable training and tuning budgets, threshold/postprocessing policies, and
hardware accounting. Include a simple baseline, the strong nnU-Net baseline,
and one-factor ablations sufficient to isolate every causal attribution. Keep
the current result system-level if those controls are not run.

### P2-02 - Statistics support a bounded negative ablation, not general uncertainty claims

**Observed.** Historical RQ1 has one run per arm and no paired per-case CI or
hypothesis test. Loop206 has three selected seeds and 76 train-screen groups.
Its 10,000-resample group bootstrap averages seeds/views first, so its interval
is conditional on those selected seeds; per-seed directions are mixed. No P1
independent-arm seed bootstrap, all-nine contrast table, multiplicity policy,
or quantitative source/size/quality/corruption/tail slices exists.

**Evidence.** `paper/clean_v3_loop206/main.tex`;
`paper/clean_v3_loop206/sections/05_experiments.tex`;
`paper/clean_v3_loop206/sections/06_results.tex`;
`paper/clean_v3_loop206/sections/08_limitations_ethics.tex`;
`demo/data/evidence_registry.json`;
`.superpowers/sdd/tasks8-13-readiness-audit.md`.

**Acceptance criterion.** Keep RQ1 as descriptive until repeated independent
arm runs and aligned per-case data exist. For P1, report all nine crossed
contrasts, independent resampling of arm seeds plus split groups, prespecified
interval/decision/multiplicity rules, complete seed results including negative
ones, and the six preregistered error-slice families. For Loop206, state
"conditional on the selected seeds" wherever the interval appears.

### P2-03 - Historical result verification is digest-bound, not clone-runnable reproduction

**Observed.** The strict local sources currently match registry hashes, but the
compact Loop191/192/206 reports and legacy source tables are not tracked in the
release. Historical configs, implementation/postprocessing modules, paired
predictions, five additional Loop206 configs, checkpoints, and caches remain
external or absent. The paper accurately states that a fresh clone can audit
recorded claims but cannot rerun all historical experiments.

**Evidence.** `demo/data/evidence_registry.json`;
`paper/clean_v3_loop206/sections/09_reproducibility.tex`;
`.artifacts/preprocessing_search/current_bdou_loop191_raw_rater_uncertainty_report.json`;
`.artifacts/preprocessing_search/current_bdou_loop192_nnunet_clean_v3_report.json`;
`.artifacts/preprocessing_search/current_bdou_loop206_final_closure_report.json`;
`reports/paper_revision/manuscript_readiness_audit.md`.

**Acceptance criterion.** Either publish an authorized compact reproduction
surface containing source reports, configs, preprocessing/metric code, locks,
data acquisition and checksum instructions, seed schedule, expected compute,
and receipt validation; or preserve the narrower phrase "digest-recorded;
source bytes verified only in the strict local audit." Do not call the
historical experiments clone-runnable until another clean environment reaches
the stated reproduction target.

### P2-04 - The defense question bank collapses historical evidence and blocked P1 readiness

**Observed.** The bank is cautious, but several answers are over-broad:

- Question 4 calls public-sample training exposure unverified, while the release
  manifest records both samples as included in L192 training and excluded from
  the L206 308-group fit/included in its 76-group holdout. The correct answer is
  observed manifest metadata plus unverified independent provenance beyond it.
- Question 6 treats fixed-cache origin/selection as wholly unverified, while the
  paper artifact manifest records deterministic first/middle/last selection,
  hashes, split scope, license fields, and three real-GPU receipts. Independent
  reproduction remains unverified.
- Question 18 says a complete leakage audit is absent without distinguishing
  recorded historical Clean-v3 audit evidence from the blocked prospective
  RQ1-v2 index/integrity report. The two lanes must not be merged.

**Risk.** A cautious student can still answer incorrectly by denying evidence
that exists or by confusing historical recorded evidence with prospective P1
admission.

**Evidence.** `docs/presentation/defense-question-bank.md`;
`release/imp_release_manifest.json`;
`paper/clean_v3_loop206/artifact_manifest.json`;
`paper/clean_v3_loop206/sections/03_data_protocol.tex`;
`experiments/rq1_v2/protocol.json`;
`.superpowers/sdd/completion-task5-p1-preflight-report.md`.

**Acceptance criterion.** Rewrite Questions 4, 6, and 18 with an explicit lane
in the first sentence: historical Paper RQ1, fixed L206 cache, live
reconstructed lane, or prospective RQ1-v2. Separate "observed manifest/source
record" from "independently rerun/verified." Cite the exact manifest/report for
each clause and retain one bounded next step.

### P3-01 - Defense-facing acceptance names drift between task numbers

**Observed.** Runbooks/README and historical audit material refer to a "Task 10
browser receipt," while current remediation material uses Task 7/Task 4 labels
for related E2E evidence. Task numbers are implementation chronology, not a
stable artifact identity.

**Evidence.** `demo/README.md`; `docs/runbooks/demo-operations.md`;
`.superpowers/sdd/task-14-claim-preflight.md`;
`.superpowers/sdd/task-14-followup-audit.md`.

**Acceptance criterion.** Replace defense-facing task-number references with a
stable canonical artifact name/path and schema version. Keep task numbers only
as historical provenance.

## Required lecturer questions

1. **What exact proposition does `0.9019` versus `0.8959` support?** Expected
   answer: recorded adaptive development-validation point-estimate direction
   under the legacy geometry contract. Trap: saying statistically superior,
   protected-test, or architecture-caused.
2. **Why is the 431-row partition not confirmatory?** Expected answer:
   development, checkpoint/model selection, and promotion decisions used it.
   Trap: treating the label `protected_validation` as untouched evidence.
3. **Why is nnU-Net a strong baseline but not a controlled architecture
   ablation?** Expected answer: it is a complete system with multiple changed
   variables. Trap: attributing the direction to encoder or decoder alone.
4. **What does the 256x256 to 384x384 path change?** Expected answer: it changes
   the metric geometry and can affect region/boundary values. Trap: treating
   resized binary masks as equivalent to restoring probabilities on original
   image geometry before thresholding.
5. **Why is the Loop206 Dice interval entirely below zero but still conditional?**
   Expected answer: groups are resampled after selected seeds/views are
   averaged; seed-selection uncertainty is not estimated. Trap: saying every
   seed is negative or the result rejects contour methods generally.
6. **Where are the six P1 jobs?** Expected answer: absent. The protocol lists
   seeds; no six configs/receipts/report exist. Trap: counting seed declarations
   as completed executions.
7. **What do all nine crossed contrasts add?** Expected answer: independent-arm
   seed uncertainty is not replaced by same-seed scheduling on one GPU. Trap:
   calling operational execution pairs statistically paired samples.
8. **Does the reconstructed Loop192 service reproduce the historical paper
   runtime?** Expected answer: no original-runtime equivalence is established;
   current evidence binds a reconstructed runtime only. Trap: equating model ID
   and checkpoint hash with complete runtime behavior.
9. **What does same-input mask drift show?** Expected answer: the tested current
   reconstructed configurations failed repeatability despite matched request
   bindings. Trap: generalizing that observation to the unavailable original
   runtime or inventing a root cause.
10. **Why did `ISIC_0016069` fail?** Expected answer: its lossless request exceeds
    the pinned 16 MiB transport contract and was rejected before the prescribed
    raw HTTP diagnostic. Trap: describing this as a model-inference failure.
11. **What did Presenter S actually see?** Expected answer: callbacks and local
    HTTP state, not rendered desktop/mobile UI. Trap: borrowing transient
    controller browser claims as S's own rehearsal evidence.
12. **Which shutdown passed?** Expected answer: different attempts recorded exit
    5 and a later bounded exit 0; no single canonical full E2E acceptance packet
    reconciles them. Trap: saying closed ports alone prove ordered cleanup.
13. **Which deck is authoritative?** Expected answer: none of the packaged set
    until 17-slide HTML/PPTX/PDF and the manifest agree. Trap: trusting the
    filename or release digest while ignoring byte/hash/slide-count drift.
14. **Were live public samples independent of training?** Expected answer:
    manifest metadata records L192 training exposure and L206 fit exclusion;
    therefore the masks are illustrative, not a generalization comparison.
15. **What can another lab reproduce today?** Expected answer: it can inspect
    source contracts and validate available digest-bound records; it cannot
    rerun the complete historical experiments or P1 six-job study from a fresh
    clone.
16. **What would justify a stronger scientific conclusion next?** Expected
    answer: verified index/leakage report, original-geometry matched protocol,
    simple and strong baselines, six locked jobs, independent-arm statistics,
    error slices, and a validated digest chain before any evidence promotion.

## Likely oral-defense traps

- Calling adaptive validation "protected" in the confirmatory sense.
- Converting a small point-estimate direction into a superiority statement.
- Treating a hash as a quality measurement.
- Treating reconstructed runtime identity as original-runtime equivalence.
- Calling a conditional group-bootstrap interval seed-population uncertainty.
- Saying the negative contour result applies to all boundary mechanisms.
- Saying "no leakage" rather than "no detected overlap under the recorded
  audit."
- Ignoring asymmetric training exposure in the live sample display.
- Calling `ISIC_0016069` a model failure instead of a size-boundary rejection.
- Saying the UI passed because source tests or another transient session passed.
- Selecting the latest exit 0 while omitting the failed full cleanup attempts.
- Claiming six jobs because six arm/seed slots are planned.
- Rehearsing the stale 12-slide HTML/PDF while discussing the current 17-slide
  source/PPTX.
- Calling registry validation full experiment reproduction.

## Strengths observed after findings

- The current paper source clearly separates historical RQ1, fixed L206 cache,
  live reconstructed display, and prospective RQ1-v2 evidence.
- The manuscript repeatedly limits RQ1 to adaptive, single-run, descriptive
  evidence and keeps test-v3 sealed.
- nnU-Net is an appropriate strong system-level baseline for the bounded
  question; the manuscript does not hide its system differences.
- The Loop206 result is a useful controlled negative result with a stated
  decision rule, group bootstrap, mixed seed directions, and explicit selected-
  seed limitation.
- Current slide source discloses reconstructed runtime, no canonical live
  receipt, model-specific sample exposure, and blocked P1 status.
- The 17-slide PPTX structure contains the intended six forward jumps, six Back
  jumps, and fade transitions; navigation implementation has focused security
  tests.
- The paper and readiness audit explicitly record missing fairness, subgroup,
  calibration, security, and other validation layers instead of converting
  them into claims.

## Final verdict

**BLOCKED for authoritative defense packaging and P1 evidence promotion.**

The bounded scientific narrative is defensible as a major-revision research
report: a historical adaptive-validation system comparison plus a controlled
negative train-screen ablation. Evidence strength is **Moderate** for rejecting
the tested Loop206 mechanism on its stated protocol and **Weak** for any
architecture-ranking interpretation. The current package is not authoritative
because the paper PDF is stale, the presentation PDF/HTML/manifest are 12-slide
artifacts while source/PPTX are 17 slides, Presenter S has no browser evidence,
runtime repeatability remains failed, shutdown status is unreconciled, and the
six P1 jobs do not exist.

Release decision: **do not promote new scientific numbers; do not rehearse from
the stale PDFs/HTML; preserve the current bounded wording; close P1-01 through
P1-05 before declaring package/runtime readiness.** No clinical, state-of-the-
art, live-accuracy, or original-runtime-equivalence claim is established by the
reviewed evidence.
