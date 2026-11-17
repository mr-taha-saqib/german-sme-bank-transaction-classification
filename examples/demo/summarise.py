"""Pretty-print API responses for run_demo.sh (reads JSON from stdin)."""
import json
import sys

mode = sys.argv[1]
d = json.load(sys.stdin)
if mode == "info":
    keys = ("model_version", "base_checkpoint", "n_classes", "confidence_threshold", "calibrated", "limits")
    print(json.dumps({k: d.get(k) for k in keys}, indent=2, ensure_ascii=False))
elif mode == "batch":
    print(f"items {d['n_items']}  ok {d['n_ok']}  errors {d['n_errors']}  "
          f"automatic {d['n_automatic']}  manual_review {d['n_manual_review']}")
    for r in d["results"]:
        if r["status"] == "ok":
            x = r["result"]
            print(f"  {r['transaction_id']:<24} {x['category']:<38} {x['confidence']:.3f}  {x['decision']}")
        else:
            print(f"  {r['transaction_id']:<24} ERROR: {r['error']}")
elif mode == "summary":
    print(json.dumps({k: d.get(k) for k in ("model_version", "predictions_in_window", "decisions")},
                     indent=2, ensure_ascii=False))
    drift = d.get("input_drift", {})
    print("input drift:", {k: (v.get("status") if isinstance(v, dict) else v) for k, v in drift.items()})
elif mode == "id":
    print(d["prediction_id"])
