# HISTORICAL / SUPERSEDED - Professor P Fast-Lane Audit

> **Historical snapshot only.** This report must not be treated as current release evidence. Its prior Slide 10, Slides 11-15 route, and `01 / 12` counter observations remain historical audit findings, not release assertions.
>
> **Authoritative current state.** Read status only from `paper/clean_v3_loop206/artifact_manifest.json`, `release/imp_release_manifest.json`, and `outputs/imp-lesion-evidence-defense-manifest.json`. The paper is `status=current`; the presentation is `package_state=complete` with 17 slides; both bind release digest `435606d5adc296be57405c65a9c725af3dff96c15f9aabf7ac0924d06387a264`. Live runtime, browser, and tunnel acceptance remain `unverified/blocked` as disclosed by `docs/presentation/presenter-s-transcript.md`. Any use of "current" in the historical body below refers only to the inspection snapshot, not today's release.

Date: 2026-07-23

## Scope and verdict

Question: Is the current paper/deck/demo package defensible for an oral defense, and what remains inadmissible?

Scope: read-only audit of the 17-slide HTML/PPTX/PDF package, all 17 rendered slides, the 14-page paper and all rendered pages, source/manifests, authoritative fast-release handoff, Presenter S transcript, defense question bank, and prospective RQ1-v2 plan. No service, tunnel, browser, training, private-data access, or Git mutation was performed.

Verdict: **usable for a bounded defense; not ready for an operational live-demo acceptance or a promoted RQ1-v2 scientific claim.** No P0 defect was observed. Five P1 defects require explicit disclosure or remediation before the corresponding claim can be made.

## Evidence labels

| Label | Meaning in this audit |
| --- | --- |
| **Observed** | Directly read, rendered, measured, or recorded in the inspected repository artifact. |
| **Established** | Supported by the frozen protocol, accepted statistical interpretation, or a sufficiently strong recorded experiment. |
| **Assumption** | Needed to interpret an artifact but not independently verified here. |
| **Speculation** | Plausible explanation or future hypothesis without decisive evidence. |

## Scores

### Paper rubric

| Criterion | Score (0-3) | Evidence-bounded reason |
| --- | ---: | --- |
| Problem clarity | 3 | **Observed:** leakage, geometry, robustness, and claim scope are explicit. |
| Novelty | 1 | **Observed:** contribution is primarily evidence governance and a negative ablation, not a new segmentation method. |
| Method correctness | 2 | **Observed:** protocols are mostly precise; historical system comparison and prospective admission remain distinct and incomplete. |
| Baselines | 2 | **Observed:** nnU-Net is a strong complete-system baseline; a matched simple baseline and component controls are absent. |
| Ablation | 2 | **Observed:** Loop206 isolates the fourth channel, but seed-population uncertainty and a complete endpoint table are absent. |
| Evaluation | 2 | **Observed:** overlap, boundary, distance, and stress metrics are named; RQ1 remains single-run and legacy-geometry. |
| Reproducibility | 1 | **Observed:** hashes and manifests support local audit, not full clone-runnable reproduction. |
| Limitations | 3 | **Observed:** limitations are specific, prominent, and aligned with current evidence. |
| **Total** | **16/24** | Useful paper with material weaknesses. |

### Defense package

| Criterion | Score (0-5) | Reason |
| --- | ---: | --- |
| Narrative | 4 | Strong evidence-first arc and narrow conclusion. |
| Scientific honesty | 5 | Adaptive validation, sealed partitions, negative result, and live-lane limits are explicit. |
| Visual evidence | 3 | Legible figures; Slide 10 describes qualitative evidence without showing the available lesion/mask panel. |
| Readability | 4 | No clipping or overflow observed across 17 slides; several cards remain text dense. |
| Pacing | 3 | Five consecutive challenge slides consume too much of an 11-minute defense. |
| Operational readiness | 3 | Package is current and structurally sound; browser/runtime acceptance remains blocked. |
| **Total** | **22/30** | Defense-ready only with bounded wording and an offline fallback. |

### Scientific and demo readiness

- **Historical Loop206 negative intervention:** **Moderate**, conditional on the three selected seeds and recorded train-screen protocol.
- **Historical RQ1 architecture ranking:** **Weak**, because it is adaptive, single-run, complete-system, and geometry-mismatched.
- **Overall scientific strength:** **Weak-to-Moderate**.
- **Operational demo readiness:** **2/5**. Governance is strong; runtime, browser, and cleanup acceptance are blocked.

## Current package observations

- **Observed:** `outputs/imp-lesion-evidence-defense-manifest.json` reports `package_state=complete`, 17 slides, current HTML/PPTX/PDF hashes, and release digest `d33d4c64580978994f9b87c13f9a961a0457309d33ef4f8c9e266d230f277e7a`.
- **Observed:** `paper/clean_v3_loop206/artifact_manifest.json` reports a current 14-page paper PDF; the strict paper audit reports `passed=true`.
- **Observed:** all 17 slide renders and all 14 paper-page renders were inspected. No clipping, overflow, or unreadable main text was observed.
- **Observed:** `demo_runtime/governance/imp.fast_release.handoff.v1/20260722T2110062254814Z/handoff.json` records `blocked_missing_prerequisite`, `determinism_status=blocked`, `cloudflare_status=deferred_external_dependency`, `p1_status=not_promoted`, `test_v3=false`, and `ph2=false`.
- **Observed:** the handoff binds acceptance digest `050afd71d97e53774ab3322288a50cfafa15f03a4ed028f85c2af11c214e38ff`, which belongs to `demo_runtime/acceptance/imp.dual_live.e2e.v1/20260723T033658169Z/acceptance.json`; that packet records `browser=not_run` and a blocked local prerequisite.
- **Assumption:** the independent handoff run ID is intentional governance indirection. It remains a traceability cost because the handoff and acceptance run IDs differ.

## Findings

### P1-01 - Historical Clean-v3 evidence and prospective RQ1-v2 admission remain easy to conflate

**Observed.** `paper/clean_v3_loop206/main.tex` and `paper/clean_v3_loop206/sections/03_data_protocol.tex` describe a leakage-audited historical Clean-v3 manifest with zero detected cross-split identity-group overlap. `paper/clean_v3_loop206/sections/05_experiments.tex` correctly calls prospective RQ1-v2 a separate blocked lane. `paper/clean_v3_loop206/sections/09_reproducibility.tex` then says the authorized verified index and integrity report are unavailable and the 2,008/431 counts are protocol metadata rather than verified identities. These statements can coexist only when the historical recorded audit and prospective admission audit are named every time.

**Risk.** A committee can reasonably hear the early wording as proof that current RQ1-v2 inputs are admitted, even though the prospective index and integrity report are blocked.

**Evidence:** `paper/clean_v3_loop206/main.tex`; `paper/clean_v3_loop206/sections/03_data_protocol.tex`; `paper/clean_v3_loop206/sections/05_experiments.tex`; `paper/clean_v3_loop206/sections/09_reproducibility.tex`; `experiments/rq1_v2/protocol.json`.

**Acceptance criterion.** Abstract, data protocol, reproducibility section, Slide 5, and oral answer must use two explicit labels: "historical Clean-v3 source-report audit" and "prospective RQ1-v2 verified admission." The latter remains blocked until a hash-bound index and zero-crossing integrity report pass. No prospective count or identity may be stated as verified before that gate.

### P1-02 - Defense answers Q4, Q6, and Q18 deny evidence that is already recorded

**Observed.** Q4 in `docs/presentation/defense-question-bank.md` calls public-sample training exposure unverified, while `release/imp_release_manifest.json` records both samples as included in L192 training and excluded from the L206 fit. Q6 calls fixed-cache origin and selection unverified, while `paper/clean_v3_loop206/artifact_manifest.json` records selection, hashes, license fields, and three real-GPU receipts. Q18 does not distinguish the historical Clean-v3 audit from the blocked prospective RQ1-v2 integrity gate.

**Risk.** Over-caution becomes factual error and collapses three evidence lanes.

**Evidence:** `docs/presentation/defense-question-bank.md`; `release/imp_release_manifest.json`; `paper/clean_v3_loop206/artifact_manifest.json`; `paper/clean_v3_loop206/sections/09_reproducibility.tex`.

**Acceptance criterion.** Q4 must say recorded manifest exposure is observed while independent training provenance remains unverified. Q6 must say fixed-cache provenance/selection is observed while independent reproduction remains unverified. Q18 must distinguish historical recorded audit evidence from blocked prospective RQ1-v2 admission. Automated tests must assert those first-sentence lane labels.

### P1-03 - Runtime, browser, and shutdown acceptance remains blocked

**Observed.** The authoritative handoff records `runtime_status=blocked_missing_prerequisite` and `determinism_status=blocked`. Its bound acceptance packet records `browser=not_run`. `docs/presentation/presenter-s-transcript.md` explicitly retains unverified desktop/mobile UI states and cleanup exit 5. The persisted combined suite exits 0 only because six environment-dependent checks are skipped; it does not manufacture missing runtime evidence.

**Risk.** A live mask, HTTP response, static slide inspection, or passing portable suite can be misrepresented as deterministic browser acceptance.

**Evidence:** `demo_runtime/governance/imp.fast_release.handoff.v1/20260722T2110062254814Z/handoff.json`; `demo_runtime/acceptance/imp.dual_live.e2e.v1/20260723T033658169Z/acceptance.json`; `docs/presentation/presenter-s-transcript.md`; `src/lesion_robustness/demo/nnunet_determinism.py`; `tests/demo/test_nnunet_determinism.py`.

**Acceptance criterion.** One canonical run must pass all private prerequisites, persistent A/B/A mask-byte repeatability, desktop `1440x900`, mobile `390x844`, failure clearing, recovery, navigation, focus, reduced-motion, overflow, and ordered cleanup exit 0 with ports 7860/7861/7862 closed. The immutable acceptance packet, handoff, transcript, and runbook must bind the same run and current release/package digests. Until then, present the offline package and say browser/runtime acceptance is blocked.

### P1-04 - Loop206's multiple gate failures are not independently auditable from the reported table

**Observed.** `paper/clean_v3_loop206/sections/06_results.tex` states that Loop206 failed primary improvement, Dice and boundary non-inferiority, clean Dice, precision, recall, distance, and per-corruption gates. `paper/clean_v3_loop206/tables/loop206_ablation.tex` and the main result figure expose only aggregate Dice and boundary-F1 deltas/intervals. Precision, recall, HD95, ASSD, clean-condition, per-corruption, and per-seed gate values are not all displayed with their thresholds.

**Risk.** Readers must trust a prose gate verdict they cannot reconstruct from the paper artifact.

**Evidence:** `paper/clean_v3_loop206/sections/05_experiments.tex`; `paper/clean_v3_loop206/sections/06_results.tex`; `paper/clean_v3_loop206/tables/loop206_ablation.tex`; `paper/clean_v3_loop206/figures/loop206_delta.pdf`.

**Acceptance criterion.** Add a compact supplementary table containing every gate name, threshold, observed aggregate, conditional interval where defined, pass/fail status, and all three seed directions. The prose verdict must be mechanically derivable from that table and the pinned receipt bundle.

### P1-05 - The deferred scientific plan is not executable

**Observed.** `docs/superpowers/plans/2026-07-23-deferred-scientific-completion.md` uses literal ellipses and leaves `DataRow`, `PerCaseRow`, `ConditionBatch`, `TrainingResult`, `ValidatedReceiptSet`, `TransferManifest`, `AuthenticatedAnalysisInputs`, `Rq1V2Report`, `build_parser`, and `parser_dispatch` undefined or incompletely connected. The independent approval remains `DEFERRED_PLAN: CHANGES_REQUIRED`.

**Risk.** A worker must invent schemas, row identity, checkpoint rules, serializers, CLI behavior, and bootstrap units during execution; those inventions can change the registered estimand.

**Evidence:** `docs/superpowers/plans/2026-07-23-deferred-scientific-completion.md`; `docs/superpowers/reviews/2026-07-23-fast-demo-slide-release-approval.md`; `src/lesion_robustness/research/rq1_data.py`; `src/lesion_robustness/research/rq1_protocol.py`.

**Acceptance criterion.** Replace every placeholder with defined dataclasses, exact field types, canonical JSON schemas, row-key construction, final-checkpoint policy, parser/subcommand behavior, transfer verification, independent arm-seed plus group bootstrap, report fields, exact red/green tests, commands, and expected outputs. Static scan must report no placeholders or undefined plan symbols before any six-job execution.

### P2-01 - Historical RQ1 ranks complete systems under a geometry mismatch

**Observed.** Loop192 predicts at 256x256 and is restored to a 384x384 metric canvas, while IMP operates at 384x384. The systems also differ in preprocessing, decoder, loss, augmentation, schedule, postprocessing, and selection policy.

**Evidence:** `paper/clean_v3_loop206/sections/04_methods.tex`; `paper/clean_v3_loop206/sections/05_experiments.tex`; `paper/clean_v3_loop206/sections/08_limitations_ethics.tex`.

**Acceptance criterion.** Keep the result as a descriptive complete-system comparison. Any stronger ranking must use the same identities, original-image geometry, conditions, metrics, tuning budget, seeds, and hardware accounting; any component claim requires a one-factor ablation.

### P2-02 - Three selected seeds do not estimate seed-population uncertainty

**Observed.** Loop206 averages the three selected seeds before split-group bootstrap. Per-seed directions are mixed. The interval is conditional on those seeds.

**Evidence:** `paper/clean_v3_loop206/main.tex`; `paper/clean_v3_loop206/sections/06_results.tex`; `paper/clean_v3_loop206/tables/loop206_ablation.tex`.

**Acceptance criterion.** Preserve "conditional on the three selected seeds" beside every Loop206 interval. A population-level uncertainty claim requires independently sampled runs and a prespecified seed-aware analysis.

### P2-03 - Slide 10 describes qualitative evidence without showing it

**Observed.** Slide 10 uses text cards despite `presentation/interactive/assets/qualitative-demo.png` and the audited paper figure being available. The slide label says qualitative/demo evidence, but no lesion, ground truth, or mask is visible.

**Evidence:** `presentation/interactive/content.json`; `presentation/interactive/assets/qualitative-demo.png`; `paper/clean_v3_loop206/figures/qualitative_demo.pdf`.

**Acceptance criterion.** Place one legible, authorized RGB/ground-truth/control/candidate row or crop on Slide 10. Retain "illustrative; not protected-test evidence" on the image and keep provenance/claim scope visible at normal presentation scale.

### P2-04 - Five consecutive challenge slides weaken defense pacing

**Observed.** Slides 11-15 repeat one challenge-card template. All are useful as discussion prompts, but five consecutive cards are expensive in an 11-minute defense.

**Evidence:** `presentation/interactive/content.json`; `outputs/imp-lesion-evidence-defense.pdf`.

**Acceptance criterion.** Default spoken route must move from Slide 10 to Slide 16; Slides 11-15 remain optional appendix/Q&A material. Rehearsed core delivery must finish within 11 minutes without omitting Slides 1-10, 16, or 17.

### P2-05 - Handoff and acceptance packets use different run identities

**Observed.** The authoritative handoff run is `20260722T2110062254814Z`; its acceptance digest binds run `20260723T033658169Z`. The validator permits this independent handoff model.

**Evidence:** `demo_runtime/governance/imp.fast_release.handoff.v1/20260722T2110062254814Z/handoff.json`; `demo_runtime/acceptance/imp.dual_live.e2e.v1/20260723T033658169Z/acceptance.json`; `src/lesion_robustness/demo/fast_release_handoff.py`.

**Acceptance criterion.** Add explicit `acceptance_run_id` to the handoff or require it to equal the packet's embedded run ID after validation. The UI/transcript must cite both identities if the handoff remains independently generated.

### P3 polish

- **Observed:** `presentation/interactive/index.html` initializes `01 / 12`; JavaScript later corrects it. **Acceptance:** initialize from 17 or render an empty counter until content loads.
- **Observed:** deck metadata uses `IMP Project Research Report`; paper uses generic author/affiliation/article metadata. **Acceptance:** add presenter, affiliation, supervisor/course, date, and target venue where authorized.
- **Observed:** paper page 4 table wraps awkwardly, page 9 panel labels are small, and page 14 is mostly blank because of bibliography pagination. **Acceptance:** visually reflow these pages without changing claims or artifact bindings.
- **Observed:** the live-lane non-reproduction limitation appears twice in the results flow. **Acceptance:** retain one canonical statement and cross-reference it elsewhere.

## Slide-by-slide visual audit

| Slide | Observed visual verdict | Defense action |
| ---: | --- | --- |
| 1 | Clean title and strong thesis; presenter identity absent. | Add presenter/affiliation. |
| 2 | Leakage story is clear and legible. | Keep. |
| 3 | Research questions and evidence classes separate well. | Keep. |
| 4 | Pipeline visually explains claim gates. | Rehearse left-to-right in under 45 seconds. |
| 5 | Clean-v3 split repair is readable. | Say "historical audit" explicitly. |
| 6 | Complete-system comparison warning is prominent. | Emphasize 256-to-384 geometry. |
| 7 | 0.9019 versus 0.8959 is visually clear. | Say point estimates, not superiority. |
| 8 | One-factor Loop206 intervention is clear. | Name selected seeds and train-screen scope. |
| 9 | Negative result is the strongest defense slide. | Add endpoint-table pointer. |
| 10 | Text-only qualitative/demo claim is visually under-evidenced. | Add one authorized image row. |
| 11 | Useful leakage challenge card. | Optional Q&A. |
| 12 | Useful fairness challenge card. | Optional Q&A. |
| 13 | Useful uncertainty challenge card. | Optional Q&A. |
| 14 | Useful demo-trust challenge card. | Optional Q&A. |
| 15 | Useful reproducibility challenge card. | Optional Q&A. |
| 16 | Honest reproducibility boundary is clear. | Keep in core route. |
| 17 | Narrow conclusion lands well. | End before opening optional cards. |

## Lecturer question bank

### 1. What does 0.9019 versus 0.8959 mean?

**Status: Observed.** They are historical Clean-v3 adaptive-validation robust-Dice point estimates for complete nnU-Net and IMP systems under the legacy geometry contract. They do not establish superiority.

### 2. Why is the validation result adaptive rather than confirmatory?

**Status: Observed.** The same validation partition informed development, selection, and promotion decisions. Repeated adaptation makes its final point estimates selection-optimistic.

### 3. Is nnU-Net a strong baseline or a controlled ablation?

**Status: Established.** It is a strong complete-system baseline. It is not a controlled component ablation because many pipeline factors differ.

### 4. Why does 256-to-384 geometry matter?

**Status: Established.** Resizing probabilities changes the evaluation path, especially boundary localization. Architecture alone cannot explain the recorded difference.

### 5. What exactly does the Loop206 interval support?

**Status: Observed.** The robust-Dice interval supports a negative aggregate candidate-minus-control effect on the 76-group train-screen protocol, conditional on the three selected seeds.

### 6. Why mention mixed seed directions?

**Status: Observed.** One seed was slightly positive and two were negative. The aggregate result must not be described as universal across seeds.

### 7. Can every claimed Loop206 gate failure be audited from the paper table?

**Status: Observed.** No. Dice and boundary F1 are shown; the complete per-gate endpoint values and thresholds are not.

### 8. Why are six P1 jobs still missing?

**Status: Observed.** The verified index/integrity inputs, model/runtime/input manifests, six locked configs, and validated receipts have not completed the prospective lane.

### 9. Why analyze nine crossed seed contrasts?

**Status: Established.** Three independent runs per arm create 3x3 arm-pair contrasts. Reporting all nine prevents favorable pairing or cherry-picking.

### 10. Is the reconstructed nnU-Net runtime equivalent to the original?

**Status: Unverified.** Hash-bound reconstructed assets establish current runtime identity, not equivalence to the original historical execution environment.

### 11. What does deterministic drift mean here?

**Status: Observed.** Repeated identical admitted inputs under the current reconstructed runtime produced different mask bytes. Cause remains speculative.

### 12. What does the oversized sample failure prove?

**Status: Observed.** It proves a fail-closed transport admission boundary. It says nothing about model accuracy or lesion difficulty.

### 13. What did Presenter S actually observe?

**Status: Observed.** Launcher/callback success, oversize clearing, recovery, source structure, HTTP 200, and closed ports. S did not visually verify the UI in a browser.

### 14. Has browser acceptance passed?

**Status: Observed.** No. The authoritative acceptance packet records `browser=not_run`.

### 15. Has cleanup acceptance passed?

**Status: Observed.** No for Presenter S. Ordered cleanup returned exit 5, despite closed ports.

### 16. Which deck and paper are authoritative?

**Status: Observed.** The current package is bound by `outputs/imp-lesion-evidence-defense-manifest.json`; the current paper is bound by `paper/clean_v3_loop206/artifact_manifest.json`; both bind release digest `d33d4c64580978994f9b87c13f9a961a0457309d33ef4f8c9e266d230f277e7a`.

### 17. Were the public demo samples exposed during training?

**Status: Observed.** The release manifest records inclusion in L192 Clean-v3 training and exclusion from the L206 308-group fit, with membership in its 76-group holdout. Independent provenance beyond the manifest remains unverified.

### 18. How was the fixed-cache qualitative row selected?

**Status: Observed.** The artifact manifest records first, middle, and last after sorting all 76 train-screen rows by sample ID and group key, without prediction or metric inspection.

### 19. What can another lab reproduce now?

**Status: Observed.** It can audit tracked source/manifests and current hashes. It cannot rerun all historical experiments from a fresh clone without external data, weights, caches, and source reports.

### 20. Did Clean-v3 pass leakage audit or not?

**Status: Observed.** Historical source-report audit evidence records zero detected cross-split identity-group overlap. Prospective RQ1-v2 verified admission remains blocked. Those are separate lanes.

### 21. Why do test-v3 and PH2 remain sealed?

**Status: Established.** Sealing prevents adaptive selection on protected partitions. No fast-lane artifact authorizes opening them.

### 22. What do the hashes prove?

**Status: Established.** They bind exact bytes and identities. They do not prove accuracy, fairness, clinical validity, runtime equivalence, or deploy readiness.

### 23. Do attractive masks prove accuracy?

**Status: Established.** No. They illustrate behavior only; representative locked ground truth and quantitative evaluation are required for accuracy claims.

### 24. What is the next defensible experiment?

**Status: Assumption.** After verified admission, run exactly three independent seeds per complete-system arm under shared original-image conditions and metric code, validate six receipts, report all nine crossed contrasts, then use independent arm-seed plus split-group bootstrap. This is the registered next step, not a completed result.

## Defense recommendation

Use Slides 1-10, 16, and 17 as the core route. Keep Slides 11-15 for questions. Run the presentation offline. State before any demo: runtime/browser acceptance is blocked, live outputs are illustrative, and neither test-v3 nor PH2 was opened. If challenged on a missing value, answer "not established by the current artifact" rather than infer it.
