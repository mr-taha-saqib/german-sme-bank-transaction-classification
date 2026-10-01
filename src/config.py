"""Central experiment configuration.

Every knob that affects a result lives here so that a run can be described by one
serialisable object. `ExperimentConfig.to_dict()` is written verbatim into
`results/<run>/config.json`, which is what makes a run reproducible.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Model registry. Keys are the short names used on the command line.
MODEL_REGISTRY: dict[str, str] = {
    # PRIMARY: German financial BERT (further-pretrained on German financial text).
    "germanfinbert": "scherrmann/GermanFinBert_FP",
    # COMPARISON: the original English FinBERT used by the reference paper.
    "finbert": "ProsusAI/finbert",
}


@dataclass
class DataConfig:
    """Where the data lives and which columns matter."""

    raw_xlsx: str = "data/synthetic_b2b_transactions_2000.xlsx"
    # Augmentation corpus. Only files carrying a `source_text` provenance column can
    # be leakage-filtered mechanically, so only those are accepted by default.
    synthetic_csvs: list[str] = field(
        default_factory=lambda: ["data/final_synthetic_clean_7033.csv"]
    )
    text_column: str = "description"
    label_column: str = "categoryName"
    parent_label_column: str = "parentCategoryName"
    # Drop rows whose cleaned text is the empty placeholder.
    drop_empty_after_clean: bool = True
    # Remove rows the SOURCE SYSTEM itself flags as duplicates (`duplicate == true`,
    # grouped by `duplicateGroupId`), keeping one representative per flagged group.
    # These pairs sometimes carry DIFFERENT descriptions (e.g. "Software Lizenz
    # 2025-02" and "SaaS Subscription 2025-02", same amount, same category), so text
    # grouping alone would let one land in train and the other in test.
    drop_source_flagged_duplicates: bool = True
    # Remove rows sharing an identical `fingerprint` (exact duplicate records).
    drop_fingerprint_duplicates: bool = True


@dataclass
class PreprocessingConfig:
    """German-aware, deliberately conservative normalisation.

    Defaults follow the brief: expand abbreviations (adds information), but do NOT
    remove legal forms or domain stop words (removes information).
    """

    lowercase: bool = True
    strip_dates: bool = True
    strip_reference_numbers: bool = True
    strip_long_digit_runs: bool = True
    expand_abbreviations: bool = True
    remove_legal_forms: bool = False
    remove_domain_stopwords: bool = False
    # Alphanumeric / punctuated tokens that must survive the digit and punctuation
    # rules. Purely alphabetic terms (kfz, it, eu) need no protection and are left
    # out so they do not shadow the legal-form and stop-word dictionaries.
    protected_tokens: list[str] = field(
        default_factory=lambda: [
            "b2b", "b2c", "p2p", "24/7", "365", "4.0", "5g", "3d", "o2",
            "q1", "q2", "q3", "q4", "h1", "h2",
        ]
    )
    empty_placeholder: str = "nodescription"


@dataclass
class GroupingConfig:
    """How near-identical descriptions are collapsed into leakage units."""

    # exact      : identical cleaned string
    # token_set  : identical *set* of tokens (order-insensitive)
    # near_dup   : token_set + label-free union-find merge of key pairs whose
    #              token-Jaccard reaches `near_duplicate_threshold` -- default
    mode: str = "near_dup"
    # 0.60 chosen from the measured similarity structure of this corpus:
    # >=0.7 merges nothing, >=0.6 merges 61 pairs with 0 crossing a category
    # boundary, >=0.5 starts fusing categories. 0.55 gives an identical grouping.
    near_duplicate_threshold: float = 0.60
    # Cross-split leakage AUDIT threshold. Must be <= the merge threshold, otherwise
    # the audit cannot detect anything the merge did not already remove.
    audit_jaccard_threshold: float = 0.60
    # Thresholds reported by the near-duplicate census in the dataset summary.
    census_thresholds: list[float] = field(
        default_factory=lambda: [0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5]
    )


@dataclass
class SplitConfig:
    """Leakage-safe, group-based train / validation / test allocation.

    Allocation is per category, on the category's list of distinct groups:
        >= 3 groups -> 1 test, 1 validation, rest train
           2 groups -> 1 test, 0 validation, 1 train
           1 group  -> 0 test, 0 validation, 1 train

    A 4-way train/validation/calibration/test split is not affordable on this corpus:
    only 13 of 37 categories have >= 4 groups. The validation split therefore also
    serves as the held-out calibration set in Step 3 -- it is disjoint from test, and
    with `training.model_selection = "fixed_epochs"` it is used for nothing else.
    """

    # "category_balanced"   -- self-contained rule, independent of the synthetic
    #                          corpus. Principled default.
    # "provenance_aligned"  -- test set = the patterns the LLM generator never saw
    #                          (i.e. those absent from the corpus's `source_text`).
    #                          Same leakage guarantee, but keeps 100% of the shipped
    #                          augmentation usable. Reported as a sensitivity check.
    strategy: str = "category_balanced"
    seed: int = 42
    min_groups_for_test: int = 2
    min_groups_for_validation: int = 3
    # If True, also build a leaky row-level random split as a labelled reference run.
    build_leaky_reference: bool = True
    leaky_test_size: float = 0.20


@dataclass
class AugmentationConfig:
    """Use of the provided LLM-generated corpus. No new generation is performed."""

    enabled: bool = True
    # Hard requirement: a synthetic row is kept only if its `source_text` maps to a
    # group that is in the TRAINING split. This is enforced at run time and does not
    # rely on the shipped file already being clean.
    require_source_in_train: bool = True
    # Drop synthetic rows whose text collides with any calibration/test pattern.
    drop_collisions_with_heldout: bool = True
    # Optional cap on synthetic examples per class (None = use all).
    max_per_class: int | None = None


@dataclass
class TrainingConfig:
    """Reference-paper hyperparameters (Table 3) with VRAM-friendly knobs added."""

    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_length: int = 128
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # "ratio" -> warmup_ratio * total_steps (safe on small corpora)
    # "fixed" -> warmup_steps literally (the paper's 500; warns if >= total steps)
    warmup_mode: str = "ratio"
    warmup_ratio: float = 0.10
    warmup_steps: int = 500
    # "focal"         -- the reference paper's weighted focal loss (gamma=2)
    # "cross_entropy" -- weighted cross-entropy, i.e. focal with gamma=0
    loss: str = "focal"
    focal_gamma: float = 2.0
    use_class_weights: bool = True
    # "fixed_epochs"    -- run exactly `epochs` epochs, select nothing. Keeps the
    #                      validation split pristine for calibration in Step 4.
    # "validation_best"  -- pick the epoch with the best validation macro-F1.
    #                      Documented trade-off: validation is then no longer unused.
    # "synthetic_dev"    -- pick on a dev slice carved from the TRAINING corpus only.
    model_selection: str = "fixed_epochs"
    synthetic_dev_fraction: float = 0.10
    seed: int = 42
    use_amp: bool = True
    num_workers: int = 0


@dataclass
class CalibrationConfig:
    """Temperature scaling fitted on a split used for nothing else."""

    # "temperature" -> single scalar T                (default, 1 parameter)
    # "vector"      -> T plus per-class bias, z/T + b (paper variant, |L|+1 params)
    method: str = "temperature"
    # "rows"   -- fit on every calibration row. Matches the deployment distribution
    #             (production sees rows), but replicas of the same pattern are not
    #             independent, so large groups dominate the fit.
    # "groups" -- fit on one averaged observation per independent pattern. Honest
    #             about the effective sample size, but far fewer points.
    # The alternative is always fitted too and recorded in model_meta.json, so the
    # sensitivity of T to this choice is visible.
    fit_on: str = "rows"
    lbfgs_lr: float = 0.05
    lbfgs_max_iter: int = 500
    ece_bins: int = 15
    # Also fit and report the paper's vector variant as a secondary result.
    also_report_vector: bool = True


@dataclass
class EvaluationConfig:
    confidence_threshold: float = 0.80
    threshold_sweep: list[float] = field(
        default_factory=lambda: [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    top_k: int = 2
    make_plots: bool = True


@dataclass
class ExperimentConfig:
    run_name: str = "main"
    output_dir: str = "results"
    models: list[str] = field(default_factory=lambda: ["germanfinbert", "finbert"])
    seeds: list[int] = field(default_factory=lambda: [42])

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            run_name=d.get("run_name", "main"),
            output_dir=d.get("output_dir", "results"),
            models=d.get("models", ["germanfinbert", "finbert"]),
            seeds=d.get("seeds", [42]),
            data=DataConfig(**d.get("data", {})),
            preprocessing=PreprocessingConfig(**d.get("preprocessing", {})),
            grouping=GroupingConfig(**d.get("grouping", {})),
            split=SplitConfig(**d.get("split", {})),
            augmentation=AugmentationConfig(**d.get("augmentation", {})),
            training=TrainingConfig(**d.get("training", {})),
            calibration=CalibrationConfig(**d.get("calibration", {})),
            evaluation=EvaluationConfig(**d.get("evaluation", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
