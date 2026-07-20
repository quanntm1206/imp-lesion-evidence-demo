# Evidence-First Paper and Live Demo Rescue Design

## Status

Approved in chat on 2026-07-20. The delivery is time-boxed to 35 hours. The
priority is a scientifically honest, complete report and a working comparison
demo, not a Q1/Q2-ready submission or a new SOTA claim.

## Goal

Deliver four connected artifacts:

1. a complete ML/CV paper draft covering background, baselines, method,
   experiments, model comparison, results, discussion, limitations, and
   conclusion;
2. a live website that compares two available models on an uploaded image and
   reports metrics only when a ground-truth mask is supplied;
3. a provenance-bound benchmark view that separates valid Clean-v3 evidence,
   train-screen evidence, and contaminated legacy evidence;
4. a clean GitHub workflow that lets the main workstation and an empty RTX 4060
   laptop divide work without committing datasets, large artifacts, secrets, or
   model weights.

## Time and Hardware Constraints

- Main workstation: RTX 5060 Ti with 16 GB VRAM and approximately 15.4 GB RAM.
- Laptop: RTX 4060 with 8 GB VRAM and 16 GB RAM; currently has no project data,
  environment, or checkpoints.
- Main workstation owns model inference, artifact verification, demo serving,
  and GPU-dependent validation.
- Laptop owns paper editing, LaTeX compilation, citation checks, UI smoke tests,
  and review from a clean clone.
- No new model panel or large retraining campaign is required inside the
  35-hour deadline.

## Evidence Classification

### Primary Clean-v3 validation comparison

The central architecture comparison uses the leakage-safe Clean-v3 validation
evidence already recorded for:

- Loop191 `L191-C0-clean-v3-IMP-control`, an IMP-SegFormer-B3 control with
  robust mean Dice `0.8958704793` and BF1 `0.4145296468`;
- Loop192 `L192-nnUNet-v2-raw-100ep`, a raw RGB nnU-Net v2 model with robust
  mean Dice `0.9019177076` and BF1 `0.4369157768`.

These are single-run validation point estimates. Loop192 did not pass the old
combined promotion gate because of BF1 and precision criteria, and protected
test-v3 was not opened. Its predictions were generated at `256x256` and scored
after resize on a `384x384` metric canvas under the older geometry contract.
The paper must disclose this limitation and must not present the comparison as
protected-test evidence, statistical superiority, or scientific SOTA.

### Controlled negative ablation

Loop206 is the causal ablation. It compares a zero fourth-channel control with
a saliency-constrained active-contour fourth-channel candidate across three
paired seeds on 76 leakage-safe Clean-v3 train-screen groups and three locked
conditions. The final closure report records:

- robust Dice delta `-0.0312962440`, paired cluster-bootstrap 95% CI
  `[-0.0491212960, -0.0156278171]`;
- BF1 delta `-0.0146583133`, 95% CI
  `[-0.0307586547, 0.0010438469]`;
- gate failure and no access to Clean-v3 validation, test-v3, or PH2.

This result supports only the bounded conclusion that the tested contour
channel harmed overlap on the preregistered train-screen protocol. It does not
prove that every contour prior or boundary-aware model is ineffective.

### Legacy evidence

Loop170 results are historical and operational only. Clean-v2 contains three
patient IDs and 13 rows crossing splits, so Loop170 is classified as
`legacy_patient_contaminated`. Its IMP, Vanilla, EGE-UNet, and nnU-Net tables
may appear only in a clearly separated legacy appendix or demo tab. They cannot
support the main conclusion, a fair Clean-v3 ranking, or an SOTA claim.

### Missing evidence

- No protected Clean-v3 test result exists.
- Loop191 and Loop192 multi-seed estimates do not exist.
- Local source images for the complete 2,869-sample Clean-v3 dataset are
  unavailable on the current Windows installation.
- Loop191 and Loop192 checkpoints are unavailable locally.
- The available Loop206 checkpoints are sufficient for live inference, but the
  demo is not a clinical system.

## Research Framing

### Research question

Under a leakage-audited dermoscopic lesion segmentation workflow, how do a
preprocessing-aware SegFormer-B3 pipeline and nnU-Net v2 trade region overlap
against boundary quality, and does an explicit saliency-constrained contour
channel improve a matched SegFormer control?

### Hypotheses

1. The self-configuring nnU-Net pipeline improves Clean-v3 validation overlap
   and boundary metrics over the existing IMP-SegFormer-B3 control.
2. The explicit contour channel improves boundary localization without a
   material Dice loss relative to its matched zero-channel control.

The recorded point estimates support the first hypothesis provisionally. The
three-seed Loop206 evidence rejects the second hypothesis for the tested
implementation and train-screen protocol.

### Claim policy

Every result is labeled by `dataset_version`, partition, evidence class,
metric contract, seed count, and source artifact. Claims must obey:

- no cross-protocol ranking;
- no statistical superiority claim from point estimates;
- no SOTA claim without protected test evidence;
- no causal preprocessing claim from Loop191 versus Loop192 because
  architecture and training pipeline both differ;
- no generalization claim from Loop206 train-screen evidence;
- no accuracy metric on an uploaded demo image without ground truth;
- no medical diagnosis, treatment, or clinical-readiness language.

## Paper Design

### Working title

`A Leakage-Aware Comparison of SegFormer and nnU-Net for Robust Skin Lesion
Segmentation with a Negative Contour-Channel Ablation`

### Section contract

1. **Abstract:** problem, leakage risk, two model families, negative ablation,
   exact evidence scope, main numbers, and limitations.
2. **Introduction:** dermoscopic segmentation motivation, robustness and
   leakage gaps, research question, and evidence-backed contributions.
3. **Related work:** SegFormer, nnU-Net, skin-lesion methods, preprocessing and
   boundary priors, leakage-safe evaluation, and segmentation metrics.
4. **Data and protocol:** Clean-v3 provenance, group-disjoint split policy,
   corruptions, partitions, metric geometry, access policy, and evidence
   classes.
5. **Methods:** IMP-SegFormer-B3, raw RGB nnU-Net v2, Loop206 zero-channel
   control, and contour-channel candidate.
6. **Experiments:** controlled variables, seeds, metrics, bootstrap procedure,
   hardware, and artifact provenance.
7. **Results:** Clean-v3 validation model comparison, three-seed Loop206
   ablation, per-condition analysis, and resource costs.
8. **Discussion:** overlap-boundary trade-off, why the contour mechanism may
   have failed, alternative explanations, and practical implications.
9. **Limitations and ethics:** missing protected test, old geometry contract,
   single-run architecture comparison, unavailable full local dataset,
   dataset licensing, and non-clinical status.
10. **Reproducibility:** code revision, configs, hashes, environment, hardware,
    checkpoint availability, and commands.
11. **Conclusion:** nnU-Net has the stronger recorded Clean-v3 validation point
    estimate; the tested contour channel is rejected; protected-test superiority
    remains unestablished.

### Tables and figures

- Evidence hierarchy and protocol table.
- Clean-v3 validation table for Loop191 versus Loop192, without significance
  markers.
- Loop206 three-seed paired ablation table with confidence intervals.
- Robustness-by-condition table where source evidence is complete.
- Architecture and evaluation pipeline diagram.
- Selected live-demo qualitative comparisons labeled as illustrative, not
  test-set evidence.
- Legacy Loop170 table in the appendix with a visible contamination warning.

All citations and numbers must resolve to a verified source or local immutable
artifact. The final paper contains no unfinished marker, fabricated citation,
unsupported claim, or unnamed data split.

## Demo Design

### User experience

The Gradio website contains three deliberately separated views:

1. **Live comparison:** upload one dermoscopic image; run one preselected
   Loop206 control checkpoint and its seed-matched contour-channel checkpoint;
   show both masks, overlays, side-by-side differences, latency, device, model
   ID, and checkpoint hash.
2. **Benchmark evidence:** show Loop191 and Loop192 Clean-v3 validation values,
   Loop206 negative-ablation statistics, protocol labels, and limitations.
3. **Legacy evidence:** show Loop170 comparisons only behind a
   `legacy_patient_contaminated` warning.

If a binary ground-truth mask is uploaded, the demo computes Dice, IoU, BF1,
HD95, and ASSD under one documented geometry contract. Without ground truth it
shows predictions and latency only. The interface must never infer clinical
meaning from the mask.

### Component boundaries

- **Model registry:** immutable model ID, config path, checkpoint path/hash,
  evidence class, seed, input contract, and display label.
- **Inference adapter:** image validation, preprocessing, model loading,
  prediction, thresholding, postprocessing, timing, and deterministic output.
- **Metric evaluator:** optional ground-truth validation and metric computation
  using one shared geometry contract.
- **Evidence registry:** normalized JSON generated only from verified local
  reports; the UI never parses arbitrary spreadsheets at request time.
- **Presentation layer:** Gradio layout, warnings, plots, tables, and exportable
  result receipt.
- **Health/provenance endpoint:** environment, CUDA availability, loaded model
  IDs, checkpoint hashes, and startup failures without secrets or local source
  paths.

### Request flow

1. Validate image type, dimensions, and decoded pixel count.
2. Normalize orientation and color mode.
3. Run the control and candidate sequentially on one GPU.
4. Restore both masks to the original uploaded geometry.
5. Compute optional metrics only after validating a same-case ground-truth
   mask.
6. Render comparison outputs and a provenance receipt.
7. Delete temporary uploads and intermediate files after the request.

### Failure handling

- Missing or hash-mismatched checkpoint blocks startup.
- CUDA failure returns a clear service error; CPU fallback is allowed only when
  explicitly enabled and labeled.
- Invalid image, oversized input, malformed mask, or geometry mismatch is
  rejected before inference.
- One model failure cannot be displayed as a valid comparison.
- NaN, infinite, or undefined metrics display `not computable` with a reason.
- Public uploads are ephemeral; filenames and paths are not logged.

### Deployment

The main workstation serves Gradio locally and exposes it through a Cloudflare
Tunnel for the demonstration window. Tunnel credentials remain outside Git.
A local-only launch remains the fallback. The service runs one inference
request at a time to protect the 16 GB system RAM and GPU memory.

## GitHub and Two-Machine Workflow

The current workspace has no usable Git repository. A fresh repository will be
initialized after this spec is approved. The recommended remote is private
during rescue because the tree contains machine-specific paths, research
artifacts, and license-sensitive provenance that require review.

Git tracks only source code, tests, configs, compact evidence manifests,
paper source, documentation, and small figures. It excludes:

- datasets and uploaded demo images;
- `.artifacts`, caches, logs, generated predictions, and virtual environments;
- `*.pt`, `*.pth`, `*.ckpt`, and other large model files;
- tunnel tokens, API keys, absolute-machine secrets, and private paths;
- generated LaTeX build files except the final review PDF when desired.

Model weights stay on the main workstation. If the laptop later needs them,
they are transferred privately after license review, not committed to Git or
Git LFS by default.

### Work split

**Main workstation**

- repair/init Git, sanitize tracked content, and create the remote-ready tree;
- verify evidence JSON and checkpoint hashes;
- create and test the Loop206 inference path;
- run GPU integration tests and latency measurements;
- host the public demo and produce final screenshots/receipts.

**RTX 4060 laptop**

- clone the clean repository;
- create an independent paper/QA branch;
- compile and proofread LaTeX;
- verify references, tables, captions, terminology, and claim-to-evidence links;
- run unit tests and UI smoke tests that do not require weights;
- merge through GitHub after review.

GitHub coordinates code and text only. It does not act as dataset storage or a
model registry.

## Delivery Schedule

- **Hours 0-3:** repository rescue, ignore policy, evidence freeze, environment
  bootstrap, clean clone check.
- **Hours 3-10:** Loop206 model-loading proof, inference adapter, metric tests,
  minimum live Gradio comparison.
- **Hours 10-18:** complete paper draft and evidence-linked result tables.
- **Hours 18-24:** demo presentation, benchmark registry, figures, qualitative
  examples, laptop paper review.
- **Hours 24-30:** integration tests, latency/resource checks, LaTeX build,
  citation and claim audit.
- **Hours 30-35:** clean-clone rehearsal, public tunnel rehearsal, final PDF,
  demo receipt, README, and contingency buffer.

## Validation Strategy

### Paper

- LaTeX builds from a clean clone with no missing files or unresolved refs.
- Automated scan finds no unfinished marker, placeholder, unsupported SOTA wording, or
  unlabeled legacy value.
- Every numeric result maps to an evidence-registry row and source hash.
- Conclusions match evidence class, partition, metric contract, and seed count.
- References are verified against primary sources; venue rank is not claimed.

### Demo

- Unit tests cover image/mask validation, geometry restoration, metrics, empty
  masks, registry validation, and checkpoint-hash rejection.
- One known sample produces deterministic masks on repeated inference.
- GPU integration test loads both selected Loop206 checkpoints and completes a
  sequential comparison without exceeding resource limits.
- No-ground-truth requests emit no accuracy metrics.
- Clean browser smoke test verifies desktop and mobile layouts.
- Tunnel rehearsal confirms the public URL, upload, inference, and cleanup path.

### Repository

- `git status` is clean after a fresh clone on the laptop.
- Secret, absolute-path, dataset, artifact, and large-file scans pass.
- Setup and launch commands work from documented prerequisites.
- Paper and demo changes are reviewable as small commits.

## Acceptance Criteria

Delivery is accepted when:

1. a complete PDF paper draft builds without placeholders and includes
   baselines, methods, model comparison, results, limitations, and conclusion;
2. the paper makes no SOTA or protected-test claim and clearly labels all three
   evidence classes;
3. the live website compares the two real Loop206 checkpoints on a user upload;
4. optional ground-truth metrics are correct and absent otherwise;
5. a public or local fallback demo launch is documented and rehearsed;
6. the laptop can clone the repository and independently build the paper and
   run non-GPU tests;
7. no dataset, weight, secret, or oversized artifact is committed.

## Out of Scope

- Q1/Q2 venue targeting or official journal-template compliance;
- new protected validation/test/PH2 access;
- retraining Loop191, Loop192, Loop206, or a broad model panel;
- recovering missing Loop191/Loop192 checkpoints or the full dataset unless an
  external copy becomes available;
- claiming medical, diagnostic, clinical, or production readiness;
- presenting Loop170 as valid scientific ranking evidence.
