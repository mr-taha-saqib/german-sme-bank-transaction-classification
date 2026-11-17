"""Send controlled traffic to a running API and print /monitoring/summary.

normal  : original HELD-OUT descriptions (validation + test rows, sent as-is): new
          patterns resembling production traffic, matching the monitoring reference
drift   : long English free-text descriptions the model has never seen (input drift)
labels  : like normal, followed by POST /feedback with the true category (delayed labels)

Descriptions are synthetic; nothing is logged by the API except numeric features.

    python scripts/simulate_traffic.py --url http://localhost:8000 --mode normal --n 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset_io import load_frozen_split  # noqa: E402

DRIFT_WORDS = ["payment", "transfer", "invoice", "customer", "refund", "subscription", "wire", "monthly",
               "international", "account", "settlement", "merchant", "card", "fee", "chargeback", "london"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["normal", "drift", "labels"], default="normal")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    frozen = load_frozen_split(ROOT / "results" / "dataset")[0]
    heldout = frozen[frozen["split"] != "train"]
    codes: dict[int, int] = {}
    with httpx.Client(base_url=args.url, timeout=30) as c:
        for i in range(0, args.n, 25):
            k = min(25, args.n - i)
            if args.mode == "drift":
                texts = [" ".join(rng.choice(DRIFT_WORDS, size=int(rng.integers(8, 14)))) + f" ref {rng.integers(1e5, 1e6)}"
                         for _ in range(k)]
                labels = [None] * k
            else:
                rows = heldout.sample(n=k, random_state=int(rng.integers(1e9)))
                texts = rows["description"].astype(str).tolist()
                labels = rows["categoryName"].tolist()
            r = c.post("/predict/batch", json={"items": [{"description": t} for t in texts]})
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
            if args.mode == "labels" and r.status_code == 200:
                for res, lab in zip(r.json()["results"], labels):
                    c.post("/feedback", json={"prediction_id": res["result"]["prediction_id"], "true_category": lab})
        s = c.get("/monitoring/summary").json()
    print(json.dumps({"mode": args.mode, "batch_status_codes": codes, "decisions": s["decisions"],
                      "input_drift": s["input_drift"], "prediction_drift": s["prediction_drift"],
                      "observed_performance": s["observed_performance"], "latency_ms": s["latency_ms"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
