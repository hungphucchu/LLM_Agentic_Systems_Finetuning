import os
import random
from typing import Dict, List, Tuple

from src.utils.project_paths import ensure_repo_on_path

ensure_repo_on_path()

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.openai_client import build_openai_client_for_judge, call_judge_json_completion
from src.utils.prompt_loader import fill_placeholders, load_prompt


def _build_judge_prompt(row_a: Dict, row_b: Dict, ckpt_a: str, ckpt_b: str) -> str:
    instr = row_a.get("instruction", "")
    inp = row_a.get("input", "")
    resp_a = row_a.get("prediction", "")
    resp_b = row_b.get("prediction", "")
    tmpl = load_prompt(os.getenv("JUDGE_PAIRWISE_PROMPT", "prompts/judge_pairwise_eval.md"))
    return fill_placeholders(
        tmpl,
        {
            "instruction": instr,
            "input": inp,
            "response_a": resp_a,
            "response_b": resp_b,
            "checkpoint_a": ckpt_a,
            "checkpoint_b": ckpt_b,
        },
    )


def _judge_system_message() -> str:
    return load_prompt(os.getenv("JUDGE_SYSTEM_PROMPT", "prompts/judge_system_message.md"))


def _pairwise(
    rows_a: List[Dict],
    rows_b: List[Dict],
    ckpt_a: str,
    ckpt_b: str,
    *,
    randomize_order: bool,
) -> List[Dict]:
    client = build_openai_client_for_judge(log_prefix="alpaca-judge")
    results: List[Dict] = []
    n = min(len(rows_a), len(rows_b))
    sys_msg = _judge_system_message()
    for i in range(n):
        ra, rb = rows_a[i], rows_b[i]

        # Swap response order to reduce ordering bias:
        # - if swapped, "Response A" will correspond to ckpt_b (and vice versa)
        swapped = False
        if randomize_order and random.random() < 0.5:
            swapped = True
            prompt = _build_judge_prompt(rb, ra, ckpt_b, ckpt_a)
        else:
            prompt = _build_judge_prompt(ra, rb, ckpt_a, ckpt_b)
        record = call_judge_json_completion(
            client,
            system_message=sys_msg,
            user_prompt=prompt,
            log_prefix="alpaca-judge",
            failure_label="Alpaca judge",
        )
        # Attach prompt id and checkpoints if not already present.
        record.setdefault("prompt_id", f"alpaca_eval_{i:05d}")
        record.setdefault("checkpoint_a", ckpt_a)
        record.setdefault("checkpoint_b", ckpt_b)
        record.setdefault("swapped", swapped)
        results.append(record)
    return results


def main() -> None:
    ckpt0 = os.getenv("CKPT0_LABEL", "ckpt0_base")
    ckpt1 = os.getenv("CKPT1_LABEL", "ckpt1_stage1")
    ckpt2 = os.getenv("STAGE2_CKPT_LABEL", "ckpt2_stage2")
    randomize_order = os.getenv("JUDGE_RANDOMIZE_ORDER", "true").lower() in ("1", "true", "yes")

    # Required: compare all pairs (0 vs 1, 1 vs 2, 0 vs 2).
    pairs: List[Tuple[str, str, str, str]] = [
        (
            f"artifacts/predictions/{ckpt0}_alpaca_eval_outputs.jsonl",
            f"artifacts/predictions/{ckpt1}_alpaca_eval_outputs.jsonl",
            ckpt0,
            ckpt1,
        ),
        (
            f"artifacts/predictions/{ckpt1}_alpaca_eval_outputs.jsonl",
            f"artifacts/predictions/{ckpt2}_alpaca_eval_outputs.jsonl",
            ckpt1,
            ckpt2,
        ),
        (
            f"artifacts/predictions/{ckpt0}_alpaca_eval_outputs.jsonl",
            f"artifacts/predictions/{ckpt2}_alpaca_eval_outputs.jsonl",
            ckpt0,
            ckpt2,
        ),
    ]

    for path_a, path_b, ckpt_a, ckpt_b in pairs:
        rows_a = read_jsonl(path_a)
        rows_b = read_jsonl(path_b)
        if not rows_a or not rows_b:
            print(f"[alpaca-judge] Skipping pair {ckpt_a} vs {ckpt_b} due to empty predictions.")
            continue
        out_path = f"artifacts/judge/alpaca_{ckpt_a}_vs_{ckpt_b}.jsonl"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        results = _pairwise(rows_a, rows_b, ckpt_a, ckpt_b, randomize_order=randomize_order)
        write_jsonl(out_path, results)
        print(f"[alpaca-judge] Wrote {len(results)} comparisons to {out_path}")


if __name__ == "__main__":
    main()
