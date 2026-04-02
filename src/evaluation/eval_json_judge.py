import json
import os
from typing import Dict, List

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
        print("[json-judge] Warning: API_KEY/UTSA_API_KEY is not set; judge calls may fail.")

    model = os.getenv("JUDGE_MODEL", os.getenv("UTSA_MODEL", "Llama-3.1-70B-Instruct-custom"))
    print(f"[json-judge] Using base_url={base_url} model={model}")
    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    client._judge_model = model  # type: ignore[attr-defined]
    return client


def _build_json_prompt(row: Dict, ckpt: str) -> str:
    instr = row.get("instruction", "")
    inp = row.get("input", "")
    pred = row.get("prediction", "")
    ref = row.get("reference", "")

    return (
        "You are an expert judge for JSON-structured outputs.\n"
        "You will see an instruction (and optional input), the model's JSON prediction, "
        "and the reference JSON.\n"
        "Score the prediction on the following dimensions (1-5, higher is better):\n"
        "- instruction_following\n"
        "- correctness\n"
        "- clarity\n"
        "- completeness\n"
        "- structured_output_validity\n"
        "- hallucination_risk\n\n"
        "Return ONLY a JSON object with this schema:\n"
        "{\n"
        '  \"prompt_id\": \"...\",\n'
        '  \"checkpoint\": \"...\",\n'
        "  \"scores\": {\n"
        "    \"instruction_following\": int,\n"
        "    \"correctness\": int,\n"
        "    \"clarity\": int,\n"
        "    \"completeness\": int,\n"
        "    \"structured_output_validity\": int,\n"
        "    \"hallucination_risk\": int\n"
        "  },\n"
        '  \"justification\": \"short natural language string\"\n'
        "}\n\n"
        "Do not include any extra keys or markdown. "
        "Do not output chain-of-thought or XML-style thinking blocks before the JSON.\n\n"
        f"Instruction: {instr}\n"
        f"Input: {inp}\n\n"
        f"Prediction (checkpoint {ckpt}):\n{pred}\n\n"
        f"Reference JSON:\n{ref}\n\n"
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
            print(f"[json-judge] retry {attempt+1}/{max_retries+1} after error: {type(e).__name__}: {e}")
            continue
        except Exception as e:  # pragma: no cover
            last_err = e
            break
    raise RuntimeError(f"json judge failed after {max_retries+1} attempts: {last_err}")


def main() -> None:
    client = _load_client()
    stage2_ckpt = os.getenv("STAGE2_CKPT_LABEL", "ckpt2_stage2")
    ckpts = os.getenv("JSON_JUDGE_CKPTS", "ckpt0_base,ckpt1_stage1," + stage2_ckpt).split(",")
    os.makedirs("artifacts/judge", exist_ok=True)

    for ck in ckpts:
        path = f"artifacts/predictions/{ck}_json_eval_outputs.jsonl"
        rows = read_jsonl(path)
        if not rows:
            print(f"[json-judge] Skipping {ck} due to empty predictions.")
            continue
        out_path = f"artifacts/judge/json_{ck}.jsonl"
        out_rows: List[Dict] = []
        for i, row in enumerate(rows):
            prompt = _build_json_prompt(row, ck)
            record = _call_judge(client, prompt)
            record.setdefault("prompt_id", f"json_eval_{i:05d}")
            record.setdefault("checkpoint", ck)
            out_rows.append(record)
        write_jsonl(out_path, out_rows)
        print(f"[json-judge] Wrote {len(out_rows)} JSON-judge records to {out_path}")


if __name__ == "__main__":
    main()
