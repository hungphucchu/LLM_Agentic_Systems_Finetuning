import json
from pathlib import Path
from typing import Dict, List

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
    # We focus on Alpaca judge comparisons between ckpt1 (Stage 1) and ckpt2 (Stage 2),
    # which is the core forgetting question.
    judge_path = "artifacts/judge/alpaca_ckpt1_stage1_vs_ckpt2_stage2.jsonl"
    p = Path(judge_path)
    if not p.exists():
        print(f"[forgetting] Judge file {judge_path} not found; run eval_alpaca_judge.py first.")
        return

    alpaca_summary = _summarize_alpaca_forgetting(judge_path)
    out_dir = Path("artifacts/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forgetting_alpaca.json"
    write_jsonl(str(out_path), [alpaca_summary])
    print(f"[forgetting] Alpaca forgetting summary:\n{json.dumps(alpaca_summary, indent=2)}")
    print(f"[forgetting] Wrote {out_path}")


if __name__ == "__main__":
    main()
