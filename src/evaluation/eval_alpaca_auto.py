from transformers import AutoTokenizer

from src.utils.io_utils import read_jsonl

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    checkpoints = ["ckpt0_base", "ckpt1_stage1", "ckpt2_stage2"]
    for ckpt in checkpoints:
        rows = read_jsonl(f"artifacts/predictions/{ckpt}_alpaca_eval_outputs.jsonl")
        toks = [len(tokenizer.encode(r.get("prediction", "") or "")) for r in rows]
        avg_len = sum(toks) / len(toks) if toks else 0
        print(f"{ckpt}: avg_output_tokens={avg_len:.2f}, samples={len(rows)}")


if __name__ == "__main__":
    main()
