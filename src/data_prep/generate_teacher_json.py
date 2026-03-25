import json
import os
import time
import random
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIError, RateLimitError

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.json_schema_utils import is_valid_json


def generate_teacher_output(client: OpenAI, instruction: str, input_text: str) -> str:
    prompt = (
        "Return only valid JSON.\n"
        f"Instruction: {instruction}\n"
        f"Input: {input_text}\n"
    )
    response = client.chat.completions.create(
        model=os.getenv("TEACHER_MODEL", "Llama-3.1-70B-Instruct-custom"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=int(os.getenv("TEACHER_MAX_TOKENS", "512")),
    )
    return response.choices[0].message.content.strip()


def _sleep_backoff(attempt: int) -> None:
    # Exponential backoff with jitter to reduce synchronized retries.
    base = 2 ** attempt
    jitter = random.uniform(0, 1.0)
    time.sleep(base + jitter)


def generate_for_prompt(
    client: OpenAI,
    instruction: str,
    input_text: str,
    *,
    max_retries: int,
    max_invalid_retries: int,
) -> Optional[Dict[str, Any]]:
    # First retry on timeouts/transient API failures, then retry when output JSON is invalid.
    for attempt in range(max_retries + 1):
        try:
            text = generate_teacher_output(client, instruction, input_text)
        except APITimeoutError:
            if attempt >= max_retries:
                return None
            _sleep_backoff(attempt)
            continue
        except RateLimitError:
            if attempt >= max_retries:
                return None
            _sleep_backoff(attempt)
            continue
        except APIError:
            if attempt >= max_retries:
                return None
            _sleep_backoff(attempt)
            continue
        except Exception:
            # For unexpected failures, do not loop forever.
            return None

        # Validate JSON. If invalid, optionally retry just by regenerating.
        valid, obj = is_valid_json(text)
        if valid:
            return {"output_obj": obj, "raw_text": text}

        # Invalid JSON handling
        for invalid_attempt in range(max_invalid_retries):
            try:
                text = generate_teacher_output(client, instruction, input_text)
            except Exception:
                return None
            valid, obj = is_valid_json(text)
            if valid:
                return {"output_obj": obj, "raw_text": text}
            _sleep_backoff(invalid_attempt)

        return None

    return None


def main() -> None:
    load_dotenv()
    # Support either generic OpenAI-style env vars (BASE_URL/API_KEY/TEACHER_MODEL)
    # or the UTSA naming convention (UTSA_BASE_URL/UTSA_API_KEY/UTSA_MODEL).
    base_url = (
        os.getenv("BASE_URL")
        or os.getenv("UTSA_BASE_URL")
        or "http://10.246.100.230/v1"
    )
    api_key = os.getenv("API_KEY") or os.getenv("UTSA_API_KEY") or "EMPTY"
    if os.getenv("TEACHER_MODEL") is None and os.getenv("UTSA_MODEL") is not None:
        os.environ["TEACHER_MODEL"] = os.environ["UTSA_MODEL"]
    timeout_seconds = float(os.getenv("TEACHER_TIMEOUT_SECONDS", "120"))
    max_prompts = os.getenv("MAX_TEACHER_PROMPTS")
    max_prompts = int(max_prompts) if max_prompts else None
    max_retries = int(os.getenv("TEACHER_MAX_RETRIES", "4"))
    max_invalid_retries = int(os.getenv("TEACHER_MAX_INVALID_RETRIES", "1"))

    if api_key == "EMPTY":
        print("Warning: API_KEY is not set (API_KEY=EMPTY). Requests may fail.")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)

    pool: List[Dict] = read_jsonl("data/processed/json_prompt_pool.jsonl")
    if max_prompts is not None:
        pool = pool[:max_prompts]
    outputs: List[Dict] = []

    for idx, row in enumerate(pool):
        instruction = row["instruction"]
        input_text = row.get("input", "")

        if idx % 5 == 0:
            print(f"[teacher-gen] {idx+1}/{len(pool)} prompts...")

        result = generate_for_prompt(
            client,
            instruction,
            input_text,
            max_retries=max_retries,
            max_invalid_retries=max_invalid_retries,
        )
        if result is None:
            continue

        obj = result["output_obj"]
        outputs.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": json.dumps(obj, ensure_ascii=False),
                "task_type": row.get("task_type", "unknown"),
            }
        )

    split_idx = max(1, int(len(outputs) * 0.8))
    write_jsonl("data/processed/json_train_teacher.jsonl", outputs[:split_idx])
    write_jsonl("data/processed/json_eval.jsonl", outputs[split_idx:])
    print(f"Saved json_train_teacher={split_idx} json_eval={len(outputs) - split_idx}")


if __name__ == "__main__":
    main()
