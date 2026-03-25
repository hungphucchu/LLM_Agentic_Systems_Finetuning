import json
from pathlib import Path
from typing import List, Dict

from src.utils.io_utils import ensure_dir, read_jsonl


def dump_placeholder_outputs(checkpoint: str, rows: List[Dict], out_file: str) -> None:
    out_rows = []
    for i, row in enumerate(rows):
        out_rows.append(
            {
                "id": i,
                "checkpoint": checkpoint,
                "instruction": row.get("instruction", ""),
                "input": row.get("input", ""),
                "prediction": "",
                "reference": row.get("output", ""),
            }
        )
    ensure_dir(str(Path(out_file).parent))
    with open(out_file, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    alpaca_eval = read_jsonl("data/processed/alpaca_eval.jsonl")
    json_eval = read_jsonl("data/processed/json_eval.jsonl")

    for ckpt in ["ckpt0_base", "ckpt1_stage1", "ckpt2_stage2"]:
        dump_placeholder_outputs(
            ckpt,
            alpaca_eval,
            f"artifacts/predictions/{ckpt}_alpaca_eval_outputs.jsonl",
        )
        dump_placeholder_outputs(
            ckpt,
            json_eval,
            f"artifacts/predictions/{ckpt}_json_eval_outputs.jsonl",
        )
    print("Created placeholder prediction files for all checkpoints.")


if __name__ == "__main__":
    main()
