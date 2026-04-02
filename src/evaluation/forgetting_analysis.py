import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.io_utils import read_jsonl, write_jsonl


def _load_alpaca_judge(path: str) -> List[Dict]:
    return read_jsonl(path)


def _summarize_alpaca_forgetting(judge_path: str) -> Dict:
    rows = _load_alpaca_judge(judge_path)
    total = len(rows)
    ckpt1_wins = ckpt2_wins = ties = 0
    for r in rows:
        w = (r.get("winner") or "").lower()
        if w == "a":
            ckpt1_wins += 1
        elif w == "b":
            ckpt2_wins += 1
        else:
            ties += 1
    summary = {
        "total_pairs": total,
        "ckpt1_wins": ckpt1_wins,
        "ckpt2_wins": ckpt2_wins,
        "ties": ties,
        "ckpt1_win_rate": (ckpt1_wins / total) if total else 0.0,
        "ckpt2_win_rate": (ckpt2_wins / total) if total else 0.0,
        "tie_rate": (ties / total) if total else 0.0,
    }
    return summary


def main() -> None:
    ckpt1 = os.getenv("CKPT1_LABEL", "ckpt1_stage1")
    ckpt2 = os.getenv("STAGE2_CKPT_LABEL", "ckpt2_stage2")

    # Core forgetting question: ckpt1 (after Stage 1) vs ckpt2 (after Stage 2).
    judge_path = f"artifacts/judge/alpaca_{ckpt1}_vs_{ckpt2}.jsonl"
    p = Path(judge_path)
    if not p.exists():
        print(f"[forgetting] Judge file {judge_path} not found; run eval_alpaca_judge.py first.")
        return

    rows_judge = read_jsonl(judge_path)

    # Win-rate summary (checkpoint-aware; supports swapped response order in judge prompts).
    total = len(rows_judge)
    ckpt1_wins = 0
    ckpt2_wins = 0
    ties = 0
    for r in rows_judge:
        winner = (r.get("winner") or "").lower()
        ckpt_a = r.get("checkpoint_a")
        ckpt_b = r.get("checkpoint_b")
        if not ckpt_a or not ckpt_b:
            continue
        if winner == "tie" or winner == "ties":
            ties += 1
        elif winner == "a":
            if ckpt_a == ckpt1:
                ckpt1_wins += 1
            elif ckpt_a == ckpt2:
                ckpt2_wins += 1
        elif winner == "b":
            if ckpt_b == ckpt1:
                ckpt1_wins += 1
            elif ckpt_b == ckpt2:
                ckpt2_wins += 1
        else:
            # Unknown winner string: ignore
            continue

    win_rate_ckpt1 = ckpt1_wins / total if total else 0.0
    win_rate_ckpt2 = ckpt2_wins / total if total else 0.0
    tie_rate = ties / total if total else 0.0

    # Automatic metric deltas from alpaca_auto metrics.
    def _load_alpaca_auto_metrics(ckpt: str) -> Dict[str, Any]:
        path = f"artifacts/metrics/alpaca_auto_metrics_{ckpt}.json"
        if not Path(path).exists():
            return {}
        rows = read_jsonl(path)
        return rows[0] if rows else {}

    m1 = _load_alpaca_auto_metrics(ckpt1)
    m2 = _load_alpaca_auto_metrics(ckpt2)
    rougeL_1 = float(m1.get("rougeL_f1", 0.0) or 0.0)
    rougeL_2 = float(m2.get("rougeL_f1", 0.0) or 0.0)
    bert_1 = float(m1.get("bertscore_f1_avg", 0.0) or 0.0)
    bert_2 = float(m2.get("bertscore_f1_avg", 0.0) or 0.0)

    delta_win_ckpt2_minus_ckpt1 = win_rate_ckpt2 - win_rate_ckpt1
    delta_rougeL = rougeL_2 - rougeL_1
    delta_bert = bert_2 - bert_1

    # Category breakdown (best-effort heuristic based on instruction keywords).
    def _category_for_instruction(instr: str) -> str:
        s = (instr or "").lower()
        if "summar" in s:
            return "summarization"
        if "rewrite" in s or "rephrase" in s or "reword" in s:
            return "rewriting"
        if "brainstorm" in s or "ideas" in s:
            return "brainstorming"
        if "question" in s or "answer" in s or any(k in s for k in [" who ", " what ", " when ", " where ", " why ", " how "]):
            return "qa"
        return "open_ended"

    pred_rows_1 = read_jsonl(f"artifacts/predictions/{ckpt1}_alpaca_eval_outputs.jsonl")
    pred_rows_2 = read_jsonl(f"artifacts/predictions/{ckpt2}_alpaca_eval_outputs.jsonl")

    # Index-aligned category aggregation.
    cat_stats: Dict[str, Dict[str, int]] = {}
    regress_example: Optional[Dict[str, Any]] = None
    improve_example: Optional[Dict[str, Any]] = None

    for i, r in enumerate(rows_judge):
        instr = ""
        if i < len(pred_rows_1):
            instr = pred_rows_1[i].get("instruction", "") or ""
        cat = _category_for_instruction(instr)
        cat_stats.setdefault(cat, {"ckpt1_wins": 0, "ckpt2_wins": 0, "ties": 0, "total": 0})
        cat_stats[cat]["total"] += 1

        winner = (r.get("winner") or "").lower()
        ckpt_a = r.get("checkpoint_a")
        ckpt_b = r.get("checkpoint_b")

        winner_ckpt: Optional[str] = None
        if winner == "tie" or winner == "ties":
            winner_ckpt = None
        elif winner == "a":
            winner_ckpt = ckpt_a
        elif winner == "b":
            winner_ckpt = ckpt_b

        if winner_ckpt == ckpt1:
            cat_stats[cat]["ckpt1_wins"] += 1
            if regress_example is None and i < len(pred_rows_1) and i < len(pred_rows_2):
                regress_example = {
                    "idx": i,
                    "category": cat,
                    "instruction": pred_rows_1[i].get("instruction", ""),
                    "input": pred_rows_1[i].get("input", ""),
                    "prediction_ckpt1": pred_rows_1[i].get("prediction", ""),
                    "prediction_ckpt2": pred_rows_2[i].get("prediction", ""),
                    "winner": ckpt1,
                }
        elif winner_ckpt == ckpt2:
            cat_stats[cat]["ckpt2_wins"] += 1
            if improve_example is None and i < len(pred_rows_1) and i < len(pred_rows_2):
                improve_example = {
                    "idx": i,
                    "category": cat,
                    "instruction": pred_rows_1[i].get("instruction", ""),
                    "input": pred_rows_1[i].get("input", ""),
                    "prediction_ckpt1": pred_rows_1[i].get("prediction", ""),
                    "prediction_ckpt2": pred_rows_2[i].get("prediction", ""),
                    "winner": ckpt2,
                }
        else:
            cat_stats[cat]["ties"] += 1

    summary: Dict[str, Any] = {
        "ckpt1": ckpt1,
        "ckpt2": ckpt2,
        "total_pairs": total,
        "ckpt1_wins": ckpt1_wins,
        "ckpt2_wins": ckpt2_wins,
        "ties": ties,
        "ckpt1_win_rate": win_rate_ckpt1,
        "ckpt2_win_rate": win_rate_ckpt2,
        "tie_rate": tie_rate,
        "delta_win_rate_ckpt2_minus_ckpt1": delta_win_ckpt2_minus_ckpt1,
        "rougeL_f1_ckpt1": rougeL_1,
        "rougeL_f1_ckpt2": rougeL_2,
        "delta_rougeL_f1_ckpt2_minus_ckpt1": delta_rougeL,
        "bertscore_f1_ckpt1": bert_1,
        "bertscore_f1_ckpt2": bert_2,
        "delta_bertscore_f1_ckpt2_minus_ckpt1": delta_bert,
        "category_breakdown": cat_stats,
        "representative_regression_example": regress_example,
        "representative_improvement_example": improve_example,
        "available_metrics_files": {
            "alpaca_auto_metrics_ckpt1": f"artifacts/metrics/alpaca_auto_metrics_{ckpt1}.json",
            "alpaca_auto_metrics_ckpt2": f"artifacts/metrics/alpaca_auto_metrics_{ckpt2}.json",
        },
    }

    out_dir = Path("artifacts/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"forgetting_alpaca_{ckpt1}_vs_{ckpt2}.json"
    write_jsonl(str(out_path), [summary])
    print(f"[forgetting] Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
