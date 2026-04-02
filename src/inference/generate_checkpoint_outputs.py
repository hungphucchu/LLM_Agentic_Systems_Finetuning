import gc
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from src.training.qlora_utils import load_tokenizer
from src.utils.io_utils import ensure_dir, read_jsonl, write_jsonl

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
STAGE1_ADAPTER = os.getenv("STAGE1_ADAPTER_PATH", "artifacts/checkpoints/stage1_alpaca_adapter")
STAGE2_ADAPTER = os.getenv("STAGE2_ADAPTER_PATH", "artifacts/checkpoints/stage2_json_adapter")

CKPT0_LABEL = os.getenv("CKPT0_LABEL", "ckpt0_base")
CKPT1_LABEL = os.getenv("CKPT1_LABEL", "ckpt1_stage1")
STAGE2_CKPT_LABEL = os.getenv("STAGE2_CKPT_LABEL", "ckpt2_stage2")


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return int(raw)


def build_prompt(row: Dict) -> str:
    instr = row.get("instruction", "") or ""
    inp = row.get("input", "") or ""
    return f"Instruction: {instr}\nInput: {inp}\nResponse: "


def build_prompt_chat(row: Dict, tokenizer) -> Optional[str]:
    if not getattr(tokenizer, "chat_template", None):
        return None
    instr = (row.get("instruction", "") or "").strip()
    inp = (row.get("input", "") or "").strip()
    user = f"{instr}\n{inp}".strip()
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# --- OPTIMIZATION 1: BATCHED GENERATION ---
def generate_for_rows(
    model,
    tokenizer,
    rows: List[Dict],
    *,
    batch_size: int = 16,  # Process 16 prompts at the same time
    max_new_tokens: int = 512,
    use_chat_template: bool = False,
) -> List[str]:
    model.eval()
    device = next(model.parameters()).device
    
    # CRITICAL: Left-padding is required for batch generation in decoder-only models
    tokenizer.padding_side = "left" 
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = pad_id

    preds: List[str] = []
    
    # Iterate through the data in chunks
    for i in tqdm(range(0, len(rows), batch_size), desc="generate", leave=False):
        batch_rows = rows[i:i + batch_size]
        prompts = []
        
        for row in batch_rows:
            if use_chat_template:
                prompt = build_prompt_chat(row, tokenizer) or build_prompt(row)
            else:
                prompt = build_prompt(row)
            prompts.append(prompt)

        # Tokenize the entire batch at once
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask").to(device)
        
        in_len = int(input_ids.shape[1])
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": pad_id,
            "use_cache": True, # Speeds up generation
        }
        
        min_new = _env_int("INFERENCE_MIN_NEW_TOKENS")
        if min_new is not None and min_new > 0:
            gen_kwargs["min_new_tokens"] = min_new

        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
        
        # Decode the batch
        for j in range(len(batch_rows)):
            # With left padding, the new tokens always start exactly at in_len
            new_tokens = out[j, in_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            preds.append(text)
            
    return preds


def write_predictions(checkpoint: str, rows: List[Dict], predictions: List[str], out_file: str) -> None:
    out_rows = []
    for i, row in enumerate(rows):
        out_rows.append({
            "id": i,
            "checkpoint": checkpoint,
            "instruction": row.get("instruction", ""),
            "input": row.get("input", ""),
            "task_type": row.get("task_type", ""),
            "prediction": predictions[i] if i < len(predictions) else "",
            "reference": row.get("output", ""),
        })
    ensure_dir(str(Path(out_file).parent))
    write_jsonl(out_file, out_rows)


def unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- OPTIMIZATION 2: FP16 AND LORA MERGING ---
def load_fp16_base_model():
    """Loads the model in standard FP16 instead of slow 4-bit."""
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

def load_and_merge_adapter(adapter_path):
    """Loads base, applies LoRA, and merges them into a single fast model."""
    base = load_fp16_base_model()
    model = PeftModel.from_pretrained(base, adapter_path)
    return model.merge_and_unload() # Bakes the adapter into the base weights


def main() -> None:
    alpaca_eval = read_jsonl("data/processed/alpaca_eval.jsonl")
    json_eval = read_jsonl("data/processed/json_eval.jsonl")
    if not alpaca_eval or not json_eval:
        raise FileNotFoundError("Evaluation datasets missing.")

    max_alpaca = _env_int("INFERENCE_MAX_ALPACA")
    max_json = _env_int("INFERENCE_MAX_JSON")
    max_new_tokens = _env_int("GEN_MAX_NEW_TOKENS") or 512
    batch_size = _env_int("INFERENCE_BATCH_SIZE") or 16 # Adjust based on VRAM

    if max_alpaca is not None: alpaca_eval = alpaca_eval[: max(0, max_alpaca)]
    if max_json is not None: json_eval = json_eval[: max(0, max_json)]

    print(f"[inference] Batch Size={batch_size}")
    use_chat = os.getenv("INFERENCE_USE_CHAT_TEMPLATE", "0").lower() in ("1", "true", "yes")

    tokenizer = load_tokenizer(BASE_MODEL)

    setups: List[Tuple[str, Callable[[], torch.nn.Module]]] = [
        (CKPT0_LABEL, lambda: load_fp16_base_model()),
        (CKPT1_LABEL, lambda: load_and_merge_adapter(STAGE1_ADAPTER)),
        (STAGE2_CKPT_LABEL, lambda: load_and_merge_adapter(STAGE2_ADAPTER)),
    ]

    for ckpt, load_model in setups:
        print(f"[inference] Loading {ckpt} ...")
        model = load_model()
        try:
            alpaca_preds = generate_for_rows(
                model, tokenizer, alpaca_eval, batch_size=batch_size, max_new_tokens=max_new_tokens
            )
            write_predictions(ckpt, alpaca_eval, alpaca_preds, f"artifacts/predictions/{ckpt}_alpaca_eval_outputs.jsonl")
            
            json_preds = generate_for_rows(
                model, tokenizer, json_eval, batch_size=batch_size, max_new_tokens=max_new_tokens, use_chat_template=use_chat
            )
            write_predictions(ckpt, json_eval, json_preds, f"artifacts/predictions/{ckpt}_json_eval_outputs.jsonl")
        finally:
            unload(model)

if __name__ == "__main__":
    main()