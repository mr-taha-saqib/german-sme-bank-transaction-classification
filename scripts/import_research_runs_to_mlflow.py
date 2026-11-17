"""Record the 10 published research runs (results/step4_comparison) in MLflow.

Read-only with respect to the research results: it reads the saved configs and
metrics and logs them as MLflow runs tagged ``source=research-archive`` so they can
be compared with new runs in the MLflow UI. Nothing is retrained or recomputed.

    python scripts/import_research_runs_to_mlflow.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mlops_settings import load_mlops_config  # noqa: E402
from src.tracking import Tracker, log_training_run  # noqa: E402


def main() -> int:
    cfg = load_mlops_config()
    cfg["mlflow"]["log_weights"] = False
    cfg["mlflow"]["register_models"] = False
    tracker = Tracker(cfg)
    if not tracker.enabled:
        print("MLflow disabled; nothing to do")
        return 1
    client = tracker._mlflow.MlflowClient()
    exp = client.get_experiment_by_name(tracker.experiment)
    done = {r.data.tags.get("research_run") for r in client.search_runs(
        [exp.experiment_id], "tags.source = 'research-archive'")}
    n = 0
    for run_dir in sorted((ROOT / "results" / "step4_comparison").iterdir()):
        if not (run_dir / "config.json").exists() or run_dir.name in done:
            continue
        with tracker.run(run_name=f"research-{run_dir.name}",
                         tags={"source": "research-archive", "research_run": run_dir.name,
                               "note": "published GPU run imported read-only; weights not available"}):
            log_training_run(tracker, run_dir)
        n += 1
    print(json.dumps({"imported": n, "already_present": len(done - {None})}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
