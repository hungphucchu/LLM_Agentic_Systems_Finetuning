from src.utils.io_utils import read_jsonl


def main():
    checkpoints = ["ckpt0_base", "ckpt1_stage1", "ckpt2_stage2"]
    for ckpt in checkpoints:
        rows = read_jsonl(f"artifacts/predictions/{ckpt}_alpaca_eval_outputs.jsonl")
        avg_len = sum(len(r.get("prediction", "").split()) for r in rows) / len(rows) if rows else 0
        print(f"{ckpt}: avg_output_tokens={avg_len:.2f}, samples={len(rows)}")


if __name__ == "__main__":
    main()
