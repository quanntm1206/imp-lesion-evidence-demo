# Data Card

## Data scope

The project records a Clean-v3 dermoscopic segmentation manifest derived from
the public **ISIC 2016, 2017, and 2018** Task 1 image/mask material. Dataset
licenses and access terms remain those of the original providers; raw images
and masks are not redistributed by this repository.

Historical aggregate metadata records 2,869 rows: **2,008** training, **431**
validation, and **430** sealed test-v3 rows. These counts do not by themselves
admit data into a new experiment.

## Split and evidence boundaries

The **historical recorded audit** reports no detected cross-split identity-group
overlap for the historical Clean-v3 manifest under its recorded checks. That is
different from **prospective RQ1-v2 admission**, which requires a separately
authorized index, exact file hash, source-byte hashes, group keys, and a new
integrity receipt. The prospective index is unresolved, so its status is
`pending/unverified`.

The full split is currently **not independently reconstructable** from a clean
clone. The repository does not contain dataset bytes, the authorized
prospective index, or every historical source artifact. The recorded 2,008/431
counts are protocol metadata, not proof that a local directory has the correct
sample identities.

Earlier Clean-v2 evidence contains three patient identifiers and 13 rows that
crossed split boundaries. It is retained only as
`legacy_patient_contaminated`; it cannot support an independent
**patient-level** generalization claim.

## Partition policy

- Historical Paper RQ1 used Clean-v3 validation adaptively for development,
  checkpoint/model selection, and promotion. It is not untouched confirmatory
  evidence.
- Loop206 used 308 training groups for fitting and 76 disjoint training groups
  as a train-screen holdout.
- Protected **test-v3** stayed sealed for the reported work.
- **PH2** stayed sealed for Loop206 and is not admitted to prospective RQ1-v2.
- Live public/synthetic uploads have no dataset membership or ground truth and
  produce no accuracy metrics.

## Prospective integrity contract

RQ1-v2 accepts only a verified `imp.rq1_v2.dataset_index.v1` index that binds
each sample ID, split, group key, source dataset, root-relative image/mask
reference, raw-byte hash, decoded-RGB hash, and mask hash. Admission must reject
cross-split group/hash overlap and the specified near-duplicate gate before any
training job starts. Test-v3 denial occurs before index resolution; PH2 denial
occurs before referenced-file resolution.

No zero-leakage or fairness result is claimed for RQ1-v2 until the authorized
index and integrity report exist and validate.

## Known representation limits

The source collections are dermoscopic and may not represent clinical camera
images, all acquisition sites, all lesion types, or the full range of skin
tones and image-quality conditions. The current evidence does not quantify
subgroup balance, annotation disagreement, demographic coverage, or external
site generalization. Image corruption experiments are controlled synthetic
stress tests, not a substitute for real distribution-shift validation.

## Handling policy

Keep images, masks, private indexes, absolute paths, uploads, and derived caches
outside GitHub. Use provider-authorized data only. Transfer private artifacts
out of band with relative-path SHA-256 verification. Never place identifying or
confidential clinical images in the unauthenticated Quick Tunnel demo.
