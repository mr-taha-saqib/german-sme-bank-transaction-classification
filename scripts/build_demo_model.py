"""Build, package and approve a TINY test model into a registry directory.

For CI and Docker smoke tests only: a 1-layer BERT trained for a few epochs on the
frozen TRAIN split, fully offline. It is NOT the GermanFinBERT production model and
its predictions have no business meaning.

    python scripts/build_demo_model.py --registry ci-registry
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.candidate_training import build_tiny_base_model, train_run  # noqa: E402
from src.config import ExperimentConfig  # noqa: E402
from src.mlops_settings import load_mlops_config  # noqa: E402
from src.model_artifact import package_run  # noqa: E402
from src.registry import ModelRegistry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    args = ap.parse_args()
    cfg = ExperimentConfig()
    cfg.training.max_length, cfg.training.epochs = 32, 4
    cfg.training.batch_size, cfg.training.learning_rate = 32, 5e-3
    with tempfile.TemporaryDirectory() as tmp:
        base = build_tiny_base_model(Path(tmp) / "base", ROOT / "results" / "dataset")
        train_run(str(base), Path(tmp) / "run", cfg, ROOT / "results" / "dataset", augmentation_version=None,
                  short_name="tinybert-demo")
        reg = ModelRegistry(Path(args.registry).resolve())
        vid = package_run(Path(tmp) / "run", reg, load_mlops_config(apply_env=False), ROOT / "results" / "dataset",
                          extra_metadata={"demo_only": "tiny test model, not GermanFinBERT"})
        reg.approve(vid, {"reason": "tiny demo model for CI / smoke tests"})
    print(vid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
