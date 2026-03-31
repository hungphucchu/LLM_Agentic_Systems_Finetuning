import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.utils.io_utils import read_jsonl


def _load_forgetting_summary() -> Dict[str, Any]:
    path = Path("artifacts/metrics/forgetting_alpaca.json")
    if not path.exists():
        return {}
    rows = read_jsonl(str(path))
    return rows[0] if rows else {}


def _load_json_validity() -> Dict[str, float]:
    # Simple loader: derive JSON validity rates directly from json_eval predictions.
    # A more complete version could re-use eval_json_auto metrics if they are saved.
    from src.utils.json_schema_utils import is_valid_json  # local import to avoid cycles

    out: Dict[str, float] = {}
    for ck in ["ckpt0_base", "ckpt1_stage1", "ckpt2_stage2"]:
        path = Path(f"artifacts/predictions/{ck}_json_eval_outputs.jsonl")
        if not path.exists():
            continue
        rows = read_jsonl(str(path))
        total = len(rows)
        valid = 0
        for r in rows:
            ok, _ = is_valid_json(r.get("prediction", ""))
            valid += int(ok)
        out[ck] = (valid / total) if total else 0.0
    return out


def main() -> None:
    tables_dir = Path("artifacts/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    forgetting = _load_forgetting_summary()
    json_validity = _load_json_validity()

    # Table 1: minimal 3-checkpoint JSON validity summary
    t1_path = tables_dir / "json_validity_by_checkpoint.csv"
    with t1_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "json_validity"])
        for ck in ["ckpt0_base", "ckpt1_stage1", "ckpt2_stage2"]:
            w.writerow([ck, json_validity.get(ck, "")])

    # Table 2: forgetting summary (judge-based Alpaca comparison ckpt1 vs ckpt2)
    t2_path = tables_dir / "alpaca_forgetting_summary.csv"
    with t2_path.open("w", newline="", encoding="utf-8") as f:
        if forgetting:
            w = csv.writer(f)
            w.writerow(list(forgetting.keys()))
            w.writerow(list(forgetting.values()))
        else:
            f.write("no forgetting summary found; run forgetting_analysis.py first\n")

    print(f"[aggregate] Wrote {t1_path} and {t2_path}")


if __name__ == "__main__":
    main()
