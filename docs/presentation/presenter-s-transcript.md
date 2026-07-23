# Presenter S Transcript

Date: 2026-07-22. Relative timestamps preserve event order; no wall-clock
timestamps were inferred after the fact. This transcript contains no tunnel
URL, upload, receipt content, log, screenshot, mask, checkpoint, or private
path.

## Scope boundary

- **Observed** means a local launcher, loopback HTTP response, or real Gradio
  callback completed during this rehearsal.
- **Unverified** means it was not visually rendered or interacted with in a
  browser during this rehearsal. Do not turn an unverified item into a spoken
  claim.
- The live lane remains an illustrative, reconstructed-runtime workflow. It is
  not Paper RQ1 evidence, an accuracy result, original-runtime equivalence,
  clinical evidence, or deploy-readiness evidence.

## Rehearsal timeline

| Time | State | Presenter record | Boundary |
| --- | --- | --- | --- |
| T+00:00 | PREFLIGHT | Ports `7860`, `7861`, and `7862` were closed before startup. | Observed. |
| T+00:01 | SIDECAR-CHECK | Guarded sidecar check-only health completed successfully. | Observed. |
| T+00:02 | SIDECAR-LIVE | Guarded persistent sidecar health completed successfully. | Observed. |
| T+00:03 | DEMO-CHECK | Guarded Gradio preflight and dual smoke completed successfully. | Observed. |
| T+00:04 | LOOPBACK | Gradio later bound loopback `7860`; local HTTP returned `200` with the non-clinical warning. | Observed. No public tunnel opened. |
| T+00:05 | PUBLIC-SUCCESS | Select the bundled public sample. State observed through the real Gradio callback: both live arms completed; ground truth was not supplied. Say: "same current RGB, IMP first, then reconstructed nnU-Net." | Observed callback result. No receipt, image, or mask content was inspected or recorded. |
| T+00:06 | REPEAT | Repeat the same bundled public sample. Both live arms completed again. | Observed callback result. Determinism is unverified here: masks were deliberately not compared. Existing runtime evidence still records determinism as blocked. |
| T+00:07 | OVERSIZE-FAILURE | Select the bundled oversized public sample. nnU-Net became unavailable; current IMP output remained; nnU-Net output and receipt output were cleared. Say: "The demo fails closed rather than reusing a stale nnU-Net result." | Observed callback result. This is a request-size boundary, not an accuracy result. |
| T+00:08 | RECOVERY | Return to the first bundled public sample. Both live arms completed. | Observed callback result. |
| T+00:09 | DECK-STATIC | Source declares 17 slides, five complete challenge cards, pipeline targets `s05-data,s06-models,s06-models,s07-validation,s08-ablation-design,s10-demo`, and backlinks only on slides 5--10. | Observed source structure only; rendered navigation, challenge-slide layout, and the demo link are unverified. |
| T+00:10 | BROWSER-BLOCKED | Browser discovery returned no available binding; explicit in-app-browser retry was unavailable. | `1440x900` and `390x844` visual states, pipeline navigation, challenge-slide interaction, focus behavior, responsive layout, and screenshots are unverified. |
| T+00:11 | CLEANUP | Ordered cleanup returned exit `5`: Gradio descendants exceeded the stop timeout. Ports `7860`, `7861`, and `7862` were nevertheless closed; no matching Gradio launcher remained after the wait. | Cleanup acceptance is blocked because nonzero exit is incomplete cleanup. Port closure is observed, not a substitute for exit `0`. |

## Spoken guardrails

- "Paper RQ1 Loop191/192, fixed L206 cache, and live L206-to-reconstructed
  Loop192 are separate evidence lanes."
- "The live demonstration has no ground truth or accuracy metric."
- "Reconstructed nnU-Net equivalence is unverified."
- "P1 remains blocked; no claim of P0/P1 closure, clinical validity, privacy
  compliance, superiority, or deploy-readiness follows from this display."
- For any question on repeatability: "This rehearsal repeated the callback;
  it did not establish deterministic masks. Current runtime determinism remains
  blocked."

## Presenter stop condition

Do not represent browser UI states or cleanup as passed. Rehearse only the
observed callback sequence until a browser binding permits visual QA and a
fresh ordered cleanup returns exit `0` with the three ports closed.
