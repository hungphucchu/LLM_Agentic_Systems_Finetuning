import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets import load_dataset

from src.utils.io_utils import write_jsonl


def normalize_example(row: Dict) -> Dict:
    return {
        "instruction": row.get("instruction", "").strip(),
        "input": row.get("input", "").strip(),
        "output": row.get("output", "").strip(),
    }


def main() -> None:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42)

    rows: List[Dict] = [normalize_example(x) for x in ds if x.get("instruction")]
    split_idx = int(len(rows) * 0.95)
    train_rows = rows[:split_idx]
    eval_rows = rows[split_idx:]

    write_jsonl("data/processed/alpaca_train.jsonl", train_rows)
    write_jsonl("data/processed/alpaca_eval.jsonl", eval_rows)
    print(f"Saved alpaca_train={len(train_rows)} alpaca_eval={len(eval_rows)}")


if __name__ == "__main__":
    main()
