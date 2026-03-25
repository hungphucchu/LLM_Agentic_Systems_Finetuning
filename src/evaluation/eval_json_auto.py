import json
from src.utils.io_utils import read_jsonl
from src.utils.json_schema_utils import is_valid_json


def main():
    checkpoints = ["ckpt0_base", "ckpt1_stage1", "ckpt2_stage2"]
    for ckpt in checkpoints:
        rows = read_jsonl(f"artifacts/predictions/{ckpt}_json_eval_outputs.jsonl")
        total = len(rows)
        valid = 0
        for r in rows:
            ok, _ = is_valid_json(r.get("prediction", ""))
            valid += int(ok)
        rate = (valid / total) if total else 0
        print(f"{ckpt}: json_validity={rate:.4f} ({valid}/{total})")


if __name__ == "__main__":
    main()
