# Defense Question Bank

## Scope and use

This is a bounded oral-defense packet. It is not a deployment claim, a P0/P1 closure, or a replacement for the claim preflight. Each answer is limited to the status and artifacts named here.

## Evidence index

- `.superpowers/sdd/task-7b-report.md`
- `.superpowers/sdd/task-8-p1-report.md`
- `.superpowers/sdd/task-9-p1-report.md`
- `.superpowers/sdd/task-10-preflight-p1-retry.md`
- `.superpowers/sdd/task-14-claim-preflight.md`
- `outputs/imp-lesion-evidence-defense-manifest.json`
- `release/imp_release_manifest.json`

## Questions and bounded answers

### 1. Why does the live L206 demo differ from the paper's Loop191/192 result?

**Status: Observed.** The paper evidence concerns RQ1 at Loop191/192. The live L206 path is a later, reconstructed demonstration path and must not be presented as the same experimental result. The distinction is recorded in the claim preflight and evidence manifests.

**Evidence:** `.superpowers/sdd/task-14-claim-preflight.md`; `outputs/imp-lesion-evidence-defense-manifest.json`.

**Acceptance next step:** Preserve separate labels for Paper RQ1 Loop191/192, fixed L206 cache, and live L206-to-reconstructed-Loop192 in every slide and spoken claim.

### 2. Is reconstructed nnU-Net equivalent to the paper model?

**Status: Unverified.** Reconstruction may reproduce an interface or workflow, but equivalence of weights, preprocessing, training state, and performance has not been established by the available evidence.

**Evidence:** `.superpowers/sdd/task-14-claim-preflight.md`; `release/imp_release_manifest.json`.

**Acceptance next step:** Establish equivalence with versioned model artifacts, exact configuration, controlled inputs, and a documented comparison against the paper pipeline.

### 3. Does a matching hash prove model accuracy or clinical validity?

**Status: Observed.** A hash can identify an artifact or output byte sequence. It does not measure segmentation accuracy, generalization, safety, or clinical utility.

**Evidence:** `outputs/imp-lesion-evidence-defense-manifest.json`; `release/imp_release_manifest.json`.

**Acceptance next step:** Add evaluation against held-out ground truth with predefined metrics and acceptance thresholds before making accuracy claims.

### 4. **Lane: live reconstructed lane - Observed manifest metadata; independently verified training provenance remains unverified.**

**Question:** Were public demonstration samples exposed during training?

**Status: Unverified.** The available packet does not establish full training provenance or exclusion of every public sample. No claim of training-data separation is made.

**Evidence:** `.superpowers/sdd/task-14-claim-preflight.md`; `release/imp_release_manifest.json`.

**Acceptance next step:** Publish dataset provenance, split definitions, sample identifiers or privacy-safe hashes, and a leakage audit.

### 5. Why are there no ground-truth overlays or quantitative metrics in the live demo?

**Status: Blocked.** The cited live evidence is not paired with validated ground truth and metrics. Therefore the demo shows a workflow/output only, not measured segmentation quality.

**Evidence:** `.superpowers/sdd/task-9-p1-report.md`; `.superpowers/sdd/task-14-claim-preflight.md`.

**Acceptance next step:** Obtain authorized reference labels, freeze an evaluation protocol, calculate metrics, and attach results to the evidence manifest.

### 6. **Lane: fixed L206 cache - Observed manifest/source records; independent reproduction remains unverified.**

**Question:** Can a fixed L206 cache introduce leakage or selection bias?

**Status: Unverified.** A fixed cache can improve repeatability, but it may also conceal sample selection, stale outputs, or overlap with development data. The available evidence does not eliminate those risks.

**Evidence:** `.superpowers/sdd/task-7b-report.md`; `.superpowers/sdd/task-14-claim-preflight.md`.

**Acceptance next step:** Record cache origin, selection criteria, content hashes, split membership, and an independent leakage review.

### 7. What uncertainty remains from a single successful run?

**Status: Unverified.** One run supports only that observed run. It does not establish repeatability across inputs, environments, random states, hardware, or operational conditions.

**Evidence:** `.superpowers/sdd/task-8-p1-report.md`; `outputs/imp-lesion-evidence-defense-manifest.json`.

**Acceptance next step:** Run a preregistered repeatability matrix with multiple inputs and environments; report failures and variability.

### 8. How do 256 and 384 geometry affect the result?

**Status: Unverified.** Geometry changes can alter resampling, field of view, voxel correspondence, memory use, and predicted masks. The cited materials do not validate equivalence between 256 and 384 paths.

**Evidence:** `.superpowers/sdd/task-10-preflight-p1-retry.md`; `.superpowers/sdd/task-14-claim-preflight.md`.

**Acceptance next step:** Define geometry-specific preprocessing, test each against ground truth, and report geometry-stratified metrics.

### 9. How does the system fail closed when output is stale?

**Status: Blocked.** The packet does not establish an end-to-end fail-closed guarantee for stale output. A stale artifact must not be represented as fresh inference without verifiable provenance.

**Evidence:** `.superpowers/sdd/task-10-preflight-p1-retry.md`; `outputs/imp-lesion-evidence-defense-manifest.json`.

**Acceptance next step:** Enforce freshness checks linking request ID, input hash, model hash, timestamp, and output hash; reject mismatches visibly.

### 10. What is known about Cloudflare privacy and data handling?

**Status: Unverified.** The evidence packet does not prove a complete Cloudflare data-flow, retention, access-control, or contractual privacy assessment. No privacy-compliance claim is made.

**Evidence:** `.superpowers/sdd/task-14-claim-preflight.md`; `release/imp_release_manifest.json`.

**Acceptance next step:** Produce an approved data-flow diagram, retention policy, access log policy, vendor configuration record, and privacy review.

### 11. Is an RTX 4060 enough capacity for the claimed use?

**Status: Unverified.** An RTX 4060 may execute a particular demonstrated workload, but the packet does not establish capacity, latency, concurrency, reliability, or safety for broader use.

**Evidence:** `.superpowers/sdd/task-8-p1-report.md`; `.superpowers/sdd/task-14-claim-preflight.md`.

**Acceptance next step:** Benchmark defined workloads with memory, latency, throughput, failure-rate, and thermal limits; publish acceptance criteria.

### 12. Why might the PDF or PPTX be stale relative to current evidence?

**Status: Observed.** Presentation artifacts can lag source manifests and reports. Their wording must be checked against current evidence rather than treated as authoritative by filename or date alone.

**Evidence:** `release/imp_release_manifest.json`; `.superpowers/sdd/task-14-claim-preflight.md`.

**Acceptance next step:** Regenerate or annotate stale PDF/PPTX files from a versioned source, then record source revision and artifact hashes.

### 13. Is there an end-to-end receipt proving a request produced this output?

**Status: Blocked.** No complete receipt is established in this packet for request-to-input-to-model-to-output provenance. Therefore end-to-end execution cannot be asserted.

**Evidence:** `.superpowers/sdd/task-10-preflight-p1-retry.md`; `outputs/imp-lesion-evidence-defense-manifest.json`.

**Acceptance next step:** Emit immutable, privacy-safe receipts containing request, input, model, execution, and output identifiers at `demo_runtime/acceptance/imp.dual_live.e2e.v1/$RunId/acceptance.json` with schema `imp.dual_live.e2e.v1`, and verify the full chain.

### 14. What is the state of the unresolved RQ1-v2 index?

**Status: Blocked.** RQ1-v2 remains unresolved. It must not be substituted with Paper RQ1 Loop191/192, fixed L206 cache, or live L206-to-reconstructed-Loop192 evidence.

**Evidence:** `.superpowers/sdd/task-14-claim-preflight.md`; `.superpowers/sdd/task-9-p1-report.md`.

**Acceptance next step:** Define the RQ1-v2 index, lock its evidence set, execute the planned evaluation, and independently review the resulting claim.

### 15. Why was test-v3 not run?

**Status: Blocked.** This scope contains no test-v3 execution and makes no inference from test-v3. Absence of that run is a limitation, not evidence of success or failure.

**Evidence:** `.superpowers/sdd/task-14-claim-preflight.md`; `.superpowers/sdd/task-10-preflight-p1-retry.md`.

**Acceptance next step:** Approve a test-v3 protocol, execute it in its authorized scope, preserve raw results, and update claims only after review.

### 16. Does a successful live display demonstrate deploy-readiness?

**Status: Unverified.** A visible result demonstrates only the observed display path. It does not establish deploy-readiness, P0/P1 closure, robustness, monitoring, privacy, or clinical safety.

**Evidence:** `.superpowers/sdd/task-8-p1-report.md`; `.superpowers/sdd/task-14-claim-preflight.md`.

**Acceptance next step:** Complete the release criteria, including independent validation, provenance receipts, operational controls, and explicit closure review.

### 17. Is adaptive validation confirmatory evidence?

**Status: Unverified.** Adaptive tuning or repeated decisions informed by interim results can be useful for development, but the available packet does not establish it as preregistered confirmatory evaluation. It must not be presented as confirmatory without a locked protocol and untouched test partition.

**Evidence:** `.superpowers/sdd/task-14-claim-preflight.md`; `.superpowers/sdd/task-9-p1-report.md`.

**Acceptance next step:** Freeze hypotheses, metrics, stopping rules, and analysis before examining a sealed confirmatory partition; report every adaptation separately.

### 18. **Lane: historical Paper RQ1 versus prospective RQ1-v2 - Historical audit evidence is recorded; prospective RQ1-v2 integrity remains blocked until its frozen audit passes.**

**Question:** Has leakage been excluded through duplicate and group-level auditing?

**Status: Blocked.** The packet does not document a complete duplicate, near-duplicate, subject/group, or source-level overlap audit across splits. Leakage exclusion is therefore not established.

**Evidence:** `.superpowers/sdd/task-14-claim-preflight.md`; `.superpowers/sdd/task-7b-report.md`.

**Acceptance next step:** Run and preserve a duplicate/near-duplicate audit plus group-aware split audit; resolve every overlap before evaluation claims.

### 19. Can a complete-system result isolate the cause of an improvement?

**Status: Unverified.** A complete-system recipe can combine preprocessing, model, cache, geometry, seed, and selection decisions. The available evidence does not isolate their causal contributions, so one component cannot be credited for the whole result.

**Evidence:** `.superpowers/sdd/task-9-p1-report.md`; `.superpowers/sdd/task-14-claim-preflight.md`.

**Acceptance next step:** Run an ablation plan with one controlled factor per contrast and report all configuration differences.

### 20. Why are independent-arm seeds, nine crossed contrasts, and split-group clusters needed?

**Status: Blocked.** The evidence packet does not establish that independent-arm seeds, all nine crossed contrasts, or split-group clustered analysis were completed. Without them, dependence and split effects can confound comparisons.

**Evidence:** `.superpowers/sdd/task-14-claim-preflight.md`; `.superpowers/sdd/task-10-preflight-p1-retry.md`.

**Acceptance next step:** Predefine independent seeds per arm, execute the nine crossed contrasts, analyze by split-group clusters, and retain the full result table.

### 21. What does the contour-channel negative result actually cover?

**Status: Observed.** A negative result for a contour channel is bounded to the tested contour-channel setup, inputs, split, and metrics. It does not prove contours are generally ineffective, nor does it rule out another implementation or study design.

**Evidence:** `.superpowers/sdd/task-9-p1-report.md`; `.superpowers/sdd/task-14-claim-preflight.md`.

**Acceptance next step:** State the exact contour representation and tested protocol; replicate with a preregistered alternative only if the research question requires it.

### 22. Do attractive examples prove accuracy?

**Status: Observed.** Visually attractive examples can illustrate behavior, but they do not establish accuracy, calibration, generalization, or clinical usefulness without representative ground truth and quantitative evaluation.

**Evidence:** `outputs/imp-lesion-evidence-defense-manifest.json`; `.superpowers/sdd/task-14-claim-preflight.md`.

**Acceptance next step:** Use examples only as illustrations; pair any accuracy claim with a locked, representative ground-truth evaluation and uncertainty reporting.

### 23. Is reported latency comparable across systems or configurations?

**Status: Unverified.** Latency is comparable only when hardware, input geometry, preprocessing, transfer, warm-up, batching, cache state, and summary statistic are aligned. The packet does not establish all of those conditions.

**Evidence:** `.superpowers/sdd/task-8-p1-report.md`; `.superpowers/sdd/task-14-claim-preflight.md`.

**Acceptance next step:** Publish a latency protocol with matched conditions, raw timings, percentile summaries, and clear inclusion/exclusion boundaries.

### 24. How are sidecar, hash, and source drift handled fail-closed?

**Status: Blocked.** The available packet does not prove a complete fail-closed control for sidecar, hash, or source drift. A mismatch must invalidate the associated claim rather than be silently tolerated.

**Evidence:** `outputs/imp-lesion-evidence-defense-manifest.json`; `release/imp_release_manifest.json`; `.superpowers/sdd/task-10-preflight-p1-retry.md`.

**Acceptance next step:** Bind sidecars to source and output hashes, verify them at presentation and release time, and stop claim rendering on any mismatch.

### 25. Are artifacts outside GitHub included and independently verifiable?

**Status: Unverified.** GitHub-visible material alone may omit external files, generated artifacts, storage objects, or environment-dependent evidence. The packet does not establish independent verification of every referenced external artifact.

**Evidence:** `release/imp_release_manifest.json`; `outputs/imp-lesion-evidence-defense-manifest.json`.

**Acceptance next step:** Inventory all external artifacts with immutable locations or archived copies, hashes, access conditions, and an independent reproduction check.

### 26. Why is test-v3 sealed?

**Status: Observed.** A sealed test-v3 partition protects against selection optimism: repeated choices based on test feedback can make apparent performance overly optimistic. The packet does not use sealed test-v3 results because none are in scope.

**Evidence:** `.superpowers/sdd/task-14-claim-preflight.md`; `.superpowers/sdd/task-10-preflight-p1-retry.md`.

**Acceptance next step:** Keep test-v3 protected until protocol, model, and analysis are locked; conduct one governed evaluation and report it without iterative selection.

## Claim guardrails

- Use **Paper RQ1 Loop191/192** only for the paper-associated result.
- Use **fixed L206 cache** only for the cached artifact path.
- Use **live L206 to reconstructed Loop192** only for the live reconstructed path.
- State **RQ1-v2 unresolved** where that index is requested.
- Do not state that any P0/P1 item is closed, that the system is deploy-ready, or that the evidence establishes accuracy, privacy compliance, or clinical validity.
