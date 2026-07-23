"""Fail-closed contracts for prospective RQ1-v2 research."""

from lesion_robustness.research.rq1_data import (
    DataIntegrityReport,
    DataRow,
    Rq1Protocol,
    audit_data,
    load_protocol,
    phash63_luminance,
    read_authorized_rows,
    ssim_luminance_256,
)
from lesion_robustness.research.rq1_metrics import MetricRow, restore_probability, score
from lesion_robustness.research.rq1_protocol import (
    CONDITIONS,
    ConditionPanel,
    apply_condition,
    build_condition_panel,
    condition_seed,
    imp_input_hashes,
    load_condition_golden,
    nnunet_input_hashes,
    ordered_panel_sha256,
)

__all__ = [
    "DataIntegrityReport",
    "DataRow",
    "Rq1Protocol",
    "audit_data",
    "load_protocol",
    "phash63_luminance",
    "read_authorized_rows",
    "ssim_luminance_256",
    "MetricRow",
    "restore_probability",
    "score",
    "CONDITIONS",
    "ConditionPanel",
    "apply_condition",
    "build_condition_panel",
    "condition_seed",
    "imp_input_hashes",
    "load_condition_golden",
    "nnunet_input_hashes",
    "ordered_panel_sha256",
]
