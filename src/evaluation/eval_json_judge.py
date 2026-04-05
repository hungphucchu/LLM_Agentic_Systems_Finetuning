import os
from typing import Dict, List

from src.utils.project_paths import ensure_repo_on_path

ensure_repo_on_path()

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.openai_client import build_openai_client_for_judge, call_judge_json_completion
from src.utils.prompt_loader import fill_placeholders, load_prompt


def _build_json_prompt(row: Dict, ckpt: str) -> str:
    instr = row.get("instruction", "")
    inp = row.get("input", "")
    pred = row.get("prediction", "")
    ref = row.get("reference", "")
    tmpl = load_prompt(os.getenv("JUDGE_JSON_PROMPT", "prompts/judge_json_eval.md"))
    return fill_placeholders(
        tmpl,
        {
            "instruction": instr,
            "input": inp,
            "prediction": pred,
            "reference": ref,
            "checkpoint": ckpt,
        },
    )


def _judge_system_message() -> str:
    return load_prompt(os.getenv("JUDGE_SYSTEM_PROMPT", "prompts/judge_system_message.md"))


def main() -> None:
    client = build_openai_client_for_judge(log_prefix="json-judge")
    sys_msg = _judge_system_message()
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
            record = call_judge_json_completion(
                client,
                system_message=sys_msg,
                user_prompt=prompt,
                log_prefix="json-judge",
                failure_label="JSON judge",
            )
            record.setdefault("prompt_id", f"json_eval_{i:05d}")
            record.setdefault("checkpoint", ck)
            out_rows.append(record)
        write_jsonl(out_path, out_rows)
        print(f"[json-judge] Wrote {len(out_rows)} JSON-judge records to {out_path}")


if __name__ == "__main__":
    main()
