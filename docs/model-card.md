# Model Card

## Systems

### IMP MiT-B3 U-Net

The main project model is **IMP MiT-B3 U-Net**. It uses an ImageNet-initialized
**MiT-B3 encoder** and the `segmentation-models-pytorch` **U-Net decoder**. It is
not the original SegFormer architecture: historical `segformer_mit` identifiers
name the encoder family, not a SegFormer MLP decoder.

The historical Paper RQ1 system uses preprocessing-aware 384 x 384 inputs. The
live demonstration uses model `L206-control-s206`, the zero-channel control from
the later Loop206 lane. Those identities must not be treated as interchangeable
scientific results.

### Comparison system

Paper RQ1 records aggregate results for nnU-Net v2 model
`L192-nnUNet-v2-raw-100ep`. The website runs a **reconstructed nnU-Net** service
from pinned artifacts. Checkpoint and runtime identity checks establish the
current service identity; they do not establish original-runtime equivalence.

## Intended use

- Audit the evidence-bounded research workflow.
- Rehearse public or synthetic images in an illustrative segmentation demo.
- Study failure handling, provenance binding, and research claim boundaries.
- Rebuild the manuscript and inspect historical aggregate evidence.

The project is **not a diagnostic or clinical system**. Do not use its masks to
make medical decisions, triage patients, estimate disease risk, or replace a
qualified clinician. Arbitrary-upload output is illustrative and unscored.

## Evidence status

- Historical Paper RQ1: single-run, adaptively selected development-validation
  point estimates under an older geometry contract.
- Loop206: a matched train-screen negative contour-channel ablation conditional
  on three selected seeds; not protected-test evidence.
- Live lane: same RGB, IMP first, reconstructed nnU-Net second; no live ground
  truth or accuracy.
- Prospective RQ1-v2: `pending/unverified`; no promoted metrics.

No lane establishes statistical superiority, state of the art, deployment
readiness, or clinical validity.

## Limitations and risks

- The protected test-v3 and PH2 partitions remain sealed.
- Historical code/config and source-report closure is incomplete.
- The historical system comparison changes architecture, preprocessing,
  training, and geometry together, so it is descriptive rather than causal.
- The current package has no prospective calibration, security, robustness, or
  subgroup fairness validation across skin tone, acquisition device, site,
  lesion type, or image-quality strata.
- A visually plausible mask is not evidence of correctness.
- Reconstructed-runtime behavior, including determinism, must be established by
  a current receipt rather than inferred from a matching model name.

## Inputs and outputs

The demo accepts a single RGB image. The guarded live path normalizes that RGB
once, runs IMP before nnU-Net, and returns two binary-mask visualizations only
when their respective arm succeeds. A partial nnU-Net failure may retain the
current IMP image while clearing nnU-Net output and the dual receipt. The live
receipt contains identity, hash, geometry, latency, device, and status fields;
it contains no diagnosis or accuracy score.

## Private artifacts

Weights, datasets, masks, probability caches, raw uploads, and runtime bundles
are not distributed through GitHub. Authorized operators transfer them
privately and verify SHA-256 before launch.
