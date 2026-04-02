import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from src.utils.io_utils import read_jsonl


def _load_first_jsonl_row(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    rows = read_jsonl(str(p))
    return rows[0] if rows else {}


def _load_forgetting_summary() -> Dict[str, Any]:
    ckpt1 = os.getenv("CKPT1_LABEL", "ckpt1_stage1")
    ckpt2 = os.getenv("STAGE2_CKPT_LABEL", "ckpt2_stage2")

    stage_path = Path(f"artifacts/metrics/forgetting_alpaca_{ckpt1}_vs_{ckpt2}.json")
    path = stage_path if stage_path.exists() else Path("artifacts/metrics/forgetting_alpaca.json")
    if not path.exists():
        return {}
    rows = read_jsonl(str(path))
    return rows[0] if rows else {}


def _load_json_auto_metrics(ckpt: str) -> Dict[str, Any]:
    # Written by eval_json_auto.py (jsonl file with one metrics row).
    return _load_first_jsonl_row(f"artifacts/metrics/json_auto_metrics_{ckpt}.json")


def _load_alpaca_auto_metrics(ckpt: str) -> Dict[str, Any]:
    return _load_first_jsonl_row(f"artifacts/metrics/alpaca_auto_metrics_{ckpt}.json")


def _discover_alpaca_judge_files() -> List[str]:
    base = Path("artifacts/judge")
    if not base.exists():
        return []
    return sorted(str(p) for p in base.glob("alpaca_*_vs_*.jsonl"))


def _compute_alpaca_checkpoint_win_and_scores(checkpoint_labels: List[str]) -> Dict[str, Any]:
    """
    Aggregates judge outputs across all alpaca pairwise comparisons.

    For each judge record:
    - checkpoint_a gets response_a_scores
    - checkpoint_b gets response_b_scores
    Winner maps winner letter ('a'/'b') to checkpoint_a/checkpoint_b.
    """
    dim_keys = ["instruction_following", "correctness", "clarity", "completeness", "structured_output_validity", "hallucination_risk"]
    stats: Dict[str, Any] = {
        ck: {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "total_compares": 0,
            "score_sums": {k: 0.0 for k in dim_keys},
        }
        for ck in checkpoint_labels
    }

    judge_files = _discover_alpaca_judge_files()
    if not judge_files:
        return {"judge_files": [], "checkpoint_stats": stats, "dim_keys": dim_keys}

    for jf in judge_files:
        rows = read_jsonl(jf)
        for r in rows:
            ck_a = r.get("checkpoint_a")
            ck_b = r.get("checkpoint_b")
            if not ck_a or not ck_b:
                continue
            if ck_a not in stats or ck_b not in stats:
                # Skip checkpoints not in our requested set.
                continue

            resp_a_scores = r.get("response_a_scores") or {}
            resp_b_scores = r.get("response_b_scores") or {}
            for k in dim_keys:
                if k in resp_a_scores:
                    stats[ck_a]["score_sums"][k] += float(resp_a_scores.get(k) or 0)
                if k in resp_b_scores:
                    stats[ck_b]["score_sums"][k] += float(resp_b_scores.get(k) or 0)

            stats[ck_a]["total_compares"] += 1
            stats[ck_b]["total_compares"] += 1

            winner = (r.get("winner") or "").lower()
            if winner == "tie" or winner == "ties":
                stats[ck_a]["ties"] += 1
                stats[ck_b]["ties"] += 1
            elif winner == "a":
                stats[ck_a]["wins"] += 1
                stats[ck_b]["losses"] += 1
            elif winner == "b":
                stats[ck_b]["wins"] += 1
                stats[ck_a]["losses"] += 1

    # Convert sums to averages + rates.
    for ck in checkpoint_labels:
        total = stats[ck]["total_compares"] or 0
        wins = stats[ck]["wins"]
        losses = stats[ck]["losses"]
        ties = stats[ck]["ties"]
        stats[ck]["avg_scores"] = {
            k: (stats[ck]["score_sums"][k] / total if total else 0.0) for k in dim_keys
        }
        non_tie = wins + losses
        stats[ck]["win_rate"] = wins / non_tie if non_tie else 0.0
        stats[ck]["tie_rate"] = ties / total if total else 0.0
        stats[ck]["loss_rate"] = losses / total if total else 0.0

    return {"judge_files": _discover_alpaca_judge_files(), "checkpoint_stats": stats, "dim_keys": dim_keys}


def main() -> None:
    tables_dir = Path("artifacts/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    ck0 = os.getenv("CKPT0_LABEL", "ckpt0_base")
    ck1 = os.getenv("CKPT1_LABEL", "ckpt1_stage1")
    ck2 = os.getenv("STAGE2_CKPT_LABEL", "ckpt2_stage2")
    run_tag = os.getenv("RUN_TAG", "").strip()
    tag_suffix = f"_{run_tag}" if run_tag else ""
    checkpoint_labels = [ck0, ck1, ck2]

    forgetting = _load_forgetting_summary()

    # Load auto metrics for JSON and Alpaca.
    json_metrics = {ck: _load_json_auto_metrics(ck) for ck in checkpoint_labels}
    alpaca_metrics = {ck: _load_alpaca_auto_metrics(ck) for ck in checkpoint_labels}

    # 1) Three-checkpoint comparison table (assignment Section 4.1)
    judge_stats = _compute_alpaca_checkpoint_win_and_scores(checkpoint_labels)
    t_three_path = tables_dir / f"three_checkpoint_comparison{tag_suffix}.csv"
    with t_three_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "checkpoint",
                "alpaca_judge_win_rate",
                "rougeL_f1",
                "bertscore_f1_avg",
                "json_validity_rate",
                "schema_compliance_rate",
                "exact_match_rate",
            ]
        )
        for ck in checkpoint_labels:
            w.writerow(
                [
                    ck,
                    judge_stats["checkpoint_stats"][ck].get("win_rate", ""),
                    alpaca_metrics[ck].get("rougeL_f1", ""),
                    alpaca_metrics[ck].get("bertscore_f1_avg", ""),
                    json_metrics[ck].get("json_validity_rate", ""),
                    json_metrics[ck].get("schema_compliance_rate", ""),
                    json_metrics[ck].get("exact_match_rate", ""),
                ]
            )

    # 2) Alpaca judge win/tie + avg scores per dimension (assignment Section 4.2)
    # Write one row per checkpoint.
    t_judge_path = tables_dir / "alpaca_judge_summary_by_checkpoint.csv"
    t_judge_path = tables_dir / f"alpaca_judge_summary_by_checkpoint{tag_suffix}.csv"
    dim_keys = judge_stats.get("dim_keys") or []
    with t_judge_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "win_rate", "tie_rate", *[f"avg_{d}" for d in dim_keys]])
        for ck in checkpoint_labels:
            st = judge_stats["checkpoint_stats"][ck]
            row = [ck, st.get("win_rate", ""), st.get("tie_rate", "")]
            for d in dim_keys:
                row.append(st.get("avg_scores", {}).get(d, ""))
            w.writerow(row)

    # 3) Keep existing CSVs (for backward compatibility with your earlier run).
    t_forget_path = tables_dir / "alpaca_forgetting_summary.csv"
    t_forget_path = tables_dir / f"alpaca_forgetting_summary{tag_suffix}.csv"
    with t_forget_path.open("w", newline="", encoding="utf-8") as f:
        if forgetting:
            w = csv.writer(f)
            # Write a 2-column kv table so it's easy to read in spreadsheets.
            w.writerow(["key", "value"])
            for k, v in forgetting.items():
                w.writerow([k, json.dumps(v) if isinstance(v, (dict, list)) else v])
        else:
            f.write("no forgetting summary found; run forgetting_analysis.py first\n")

    print(f"[aggregate] wrote {t_three_path}, {t_judge_path}, {t_forget_path}")


if __name__ == "__main__":
    main()
