import gc
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm

from src.training.qlora_utils import load_4bit_model, load_tokenizer
from src.utils.io_utils import ensure_dir, read_jsonl, write_jsonl


BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
STAGE1_ADAPTER = "artifacts/checkpoints/stage1_alpaca_adapter"
STAGE2_ADAPTER = "artifacts/checkpoints/stage2_json_adapter"


def build_prompt(row: Dict) -> str:
    # Must match training format in train_stage1_alpaca / train_stage2_json.
    instr = row.get("instruction", "") or ""
    inp = row.get("input", "") or ""
    return f"Instruction: {instr}\nInput: {inp}\nResponse: "


def generate_for_rows(
    model,
    tokenizer,
    rows: List[Dict],
    *,
    max_new_tokens: int = 512,
) -> List[str]:
    model.eval()
    device = next(model.parameters()).device
    # Align pad/eos with model config (Phi-3 often uses a list of stop token ids).
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    eos_id = getattr(model.config, "eos_token_id", None)
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    gc_conf = model.generation_config
    if gc_conf.pad_token_id is None:
        gc_conf.pad_token_id = pad_id
    preds: List[str] = []
    for row in tqdm(rows, desc="generate", leave=False):
        prompt = build_prompt(row)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        new_tokens = out[0, in_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        preds.append(text)
    return preds


def write_predictions(
    checkpoint: str,
    rows: List[Dict],
    predictions: List[str],
    out_file: str,
) -> None:
    out_rows = []
    for i, row in enumerate(rows):
        out_rows.append(
            {
                "id": i,
                "checkpoint": checkpoint,
                "instruction": row.get("instruction", ""),
                "input": row.get("input", ""),
                "prediction": predictions[i] if i < len(predictions) else "",
                "reference": row.get("output", ""),
            }
        )
    ensure_dir(str(Path(out_file).parent))
    write_jsonl(out_file, out_rows)


def unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return int(raw)


def main() -> None:
    alpaca_eval = read_jsonl("data/processed/alpaca_eval.jsonl")
    json_eval = read_jsonl("data/processed/json_eval.jsonl")
    if not alpaca_eval:
        raise FileNotFoundError("data/processed/alpaca_eval.jsonl is missing or empty.")
    if not json_eval:
        raise FileNotFoundError("data/processed/json_eval.jsonl is missing or empty.")

    max_alpaca = _env_int("INFERENCE_MAX_ALPACA")
    max_json = _env_int("INFERENCE_MAX_JSON")
    max_new_tokens = _env_int("GEN_MAX_NEW_TOKENS") or 512
    if max_alpaca is not None:
        alpaca_eval = alpaca_eval[: max(0, max_alpaca)]
    if max_json is not None:
        json_eval = json_eval[: max(0, max_json)]

    if max_alpaca is not None or max_json is not None:
        print(
            f"[inference] Subset eval: alpaca_n={len(alpaca_eval)} json_n={len(json_eval)} "
            f"(set INFERENCE_MAX_ALPACA / INFERENCE_MAX_JSON unset for full eval)"
        )
    print(f"[inference] GEN_MAX_NEW_TOKENS={max_new_tokens} (lower for faster, shorter answers)")

    tokenizer = load_tokenizer(BASE_MODEL)

    setups: List[Tuple[str, Callable[[], torch.nn.Module]]] = [
        ("ckpt0_base", lambda: load_4bit_model(BASE_MODEL)),
        ("ckpt1_stage1", lambda: PeftModel.from_pretrained(load_4bit_model(BASE_MODEL), STAGE1_ADAPTER)),
        (
            "ckpt2_stage2",
            lambda: PeftModel.from_pretrained(load_4bit_model(BASE_MODEL), STAGE2_ADAPTER),
        ),
    ]

    for ckpt, load_model in setups:
        print(f"[inference] Loading {ckpt} ...")
        model = load_model()
        try:
            alpaca_preds = generate_for_rows(
                model, tokenizer, alpaca_eval, max_new_tokens=max_new_tokens
            )
            write_predictions(
                ckpt,
                alpaca_eval,
                alpaca_preds,
                f"artifacts/predictions/{ckpt}_alpaca_eval_outputs.jsonl",
            )
            json_preds = generate_for_rows(
                model, tokenizer, json_eval, max_new_tokens=max_new_tokens
            )
            write_predictions(
                ckpt,
                json_eval,
                json_preds,
                f"artifacts/predictions/{ckpt}_json_eval_outputs.jsonl",
            )
        finally:
            unload(model)

    print("Wrote artifacts/predictions/ckpt{0,1,2}_*_eval_outputs.jsonl with model generations.")


if __name__ == "__main__":
    main()
