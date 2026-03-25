from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.json_schema_utils import is_valid_json


def validate_file(input_path: str, output_path: str) -> None:
    rows = read_jsonl(input_path)
    valid_rows = []
    for row in rows:
        valid, _ = is_valid_json(row.get("output", ""))
        if valid:
            valid_rows.append(row)
    write_jsonl(output_path, valid_rows)
    print(f"{input_path}: {len(valid_rows)}/{len(rows)} valid")


def main() -> None:
    validate_file("data/processed/json_train_teacher.jsonl", "data/processed/json_train_teacher.jsonl")
    validate_file("data/processed/json_eval.jsonl", "data/processed/json_eval.jsonl")


if __name__ == "__main__":
    main()
