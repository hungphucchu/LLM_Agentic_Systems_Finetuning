import json
from typing import Dict, List

from src.utils.io_utils import write_jsonl


def build_prompt_pool() -> List[Dict]:
    # Minimal starter pool. Expand to >=100 for final experiments.
    tasks = [
        {
            "task_type": "json_extraction",
            "instruction": "Extract entities and dates into JSON.",
            "input": "Alice met Bob in San Antonio on 2024-05-12.",
            "json_example": "{\"entities\":[\"Alice\",\"Bob\"],\"location\":\"San Antonio\",\"date\":\"2024-05-12\"}"
        },
        {
            "task_type": "schema_generation",
            "instruction": "Generate a profile object matching schema.",
            "input": json.dumps({"required": ["name", "age", "city"]}),
            "json_example": "{\"name\":\"Avery\",\"age\":28,\"city\":\"Austin\"}"
        },
        {
            "task_type": "exact_label_classification",
            "instruction": "Classify sentiment into one of [positive, neutral, negative] and return JSON.",
            "input": "The product is okay, but shipping was slow.",
            "json_example": "{\"label\":\"neutral\"}"
        },
        {
            "task_type": "json_repair",
            "instruction": "Repair malformed JSON and return valid JSON only.",
            "input": '{"name":"Sam","age": 22,}',
            "json_example": "{\"name\":\"Sam\",\"age\":22}"
        },
        {
            "task_type": "tool_call_args",
            "instruction": "Create JSON arguments for function weather(city, date).",
            "input": "Get weather for Austin tomorrow.",
            "json_example": "{\"function\":\"weather\",\"arguments\":{\"city\":\"Austin\",\"date\":\"tomorrow\"}}"
        },
    ]
    return tasks


def main() -> None:
    rows = build_prompt_pool()
    write_jsonl("data/processed/json_prompt_pool.jsonl", rows)
    print(f"Saved json_prompt_pool={len(rows)}")


if __name__ == "__main__":
    main()
