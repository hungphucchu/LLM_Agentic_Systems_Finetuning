import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.json_schema_utils import is_valid_json


def generate_teacher_output(client: OpenAI, instruction: str, input_text: str) -> str:
    prompt = (
        "Return only valid JSON.\n"
        f"Instruction: {instruction}\n"
        f"Input: {input_text}\n"
    )
    response = client.chat.completions.create(
        model="Llama-3.1-70B-Instruct-custom",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    load_dotenv()
    base_url = os.getenv("BASE_URL", "http://10.246.100.230/v1")
    api_key = os.getenv("API_KEY", "EMPTY")
    client = OpenAI(base_url=base_url, api_key=api_key)

    pool: List[Dict] = read_jsonl("data/processed/json_prompt_pool.jsonl")
    outputs: List[Dict] = []

    for row in pool:
        text = generate_teacher_output(client, row["instruction"], row.get("input", ""))
        valid, obj = is_valid_json(text)
        if not valid:
            continue
        outputs.append(
            {
                "instruction": row["instruction"],
                "input": row.get("input", ""),
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
