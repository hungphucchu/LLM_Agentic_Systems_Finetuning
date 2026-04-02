import json
import os
import random
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIError, RateLimitError

from src.utils.json_schema_utils import assistant_message_text, parse_llm_json_dict
from src.utils.io_utils import read_jsonl, write_jsonl


def _load_client() -> OpenAI:
    load_dotenv()
    base_url = (
        os.getenv("BASE_URL")
        or os.getenv("UTSA_BASE_URL")
        or "http://10.246.100.230/v1"
    )
    api_key = os.getenv("API_KEY") or os.getenv("UTSA_API_KEY") or "EMPTY"

    if base_url:
        base_url = base_url.strip().replace("Links to an external site.", "").strip()
    if isinstance(api_key, str):
        api_key = api_key.strip()

    if api_key == "EMPTY":
        print("[alpaca-judge] Warning: API_KEY/UTSA_API_KEY is not set; judge calls may fail.")

    model = os.getenv("JUDGE_MODEL", os.getenv("UTSA_MODEL", "Llama-3.1-70B-Instruct-custom"))
    print(f"[alpaca-judge] Using base_url={base_url} model={model}")
    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    client._judge_model = model  # type: ignore[attr-defined]
    return client


def _build_judge_prompt(row_a: Dict, row_b: Dict, ckpt_a: str, ckpt_b: str) -> str:
    instr = row_a.get("instruction", "")
    inp = row_a.get("input", "")
    resp_a = row_a.get("prediction", "")
    resp_b = row_b.get("prediction", "")

    return (
        "You are an expert judge for instruction-following quality.\n"
        "You will see one instruction (and optional input) plus two candidate responses.\n"
        "Your job is to score each response on multiple dimensions and then pick a winner.\n\n"
        "Dimensions (1-5, higher is better):\n"
        "- instruction_following\n"
        "- correctness\n"
        "- clarity\n"
        "- completeness\n"
        "- structured_output_validity (for JSON-like outputs; otherwise use 3 as neutral)\n"
        "- hallucination_risk (1 = very hallucinated, 5 = minimal hallucination)\n\n"
        "Return ONLY a single JSON object with this schema:\n"
        "{\n"
        '  \"prompt_id\": \"...\",\n'
        '  \"checkpoint_a\": \"...\",\n'
        '  \"checkpoint_b\": \"...\",\n'
        "  \"response_a_scores\": {\n"
        "    \"instruction_following\": int,\n"
        "    \"correctness\": int,\n"
        "    \"clarity\": int,\n"
        "    \"completeness\": int,\n"
        "    \"structured_output_validity\": int,\n"
        "    \"hallucination_risk\": int\n"
        "  },\n"
        "  \"response_b_scores\": { ... same keys ... },\n"
        '  \"winner\": \"A\" | \"B\" | \"tie\",\n'
        "  \"justification\": \"short natural language string\"\n"
        "}\n\n"
        "Do not include any extra keys, comments, or markdown. "
        "Do not output chain-of-thought or XML-style thinking blocks before the JSON.\n\n"
        f"Instruction: {instr}\n"
        f"Input: {inp}\n\n"
        f"Response A (checkpoint {ckpt_a}):\n{resp_a}\n\n"
        f"Response B (checkpoint {ckpt_b}):\n{resp_b}\n\n"
        "Now return the JSON object."
    )


_JUDGE_SYSTEM = (
    "You must reply with exactly one valid JSON object and no markdown fences, "
    "no code blocks, and no text before or after the JSON. "
    "Do not use chain-of-thought, reasoning tags, or prose; output only the JSON object."
)


def _call_judge(client: OpenAI, prompt: str, max_retries: int = 3) -> Dict:
    model = getattr(client, "_judge_model")
    max_tokens = int(os.getenv("JUDGE_MAX_TOKENS", "4096"))
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            text = assistant_message_text(resp)
            if not text:
                fr = getattr(resp.choices[0], "finish_reason", None)
                raise ValueError(f"empty judge response (finish_reason={fr})")
            return parse_llm_json_dict(text)
        except (APITimeoutError, RateLimitError, APIError, json.JSONDecodeError, ValueError) as e:
            last_err = e
            print(f"[alpaca-judge] retry {attempt+1}/{max_retries+1} after error: {type(e).__name__}: {e}")
            continue
        except Exception as e:  # pragma: no cover - unexpected
            last_err = e
            break
    raise RuntimeError(f"judge failed after {max_retries+1} attempts: {last_err}")


def _pairwise(
    rows_a: List[Dict],
    rows_b: List[Dict],
    ckpt_a: str,
    ckpt_b: str,
    *,
    randomize_order: bool,
) -> List[Dict]:
    client = _load_client()
    results: List[Dict] = []
    n = min(len(rows_a), len(rows_b))
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
        record = _call_judge(client, prompt)
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
