import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.io_utils import read_jsonl, write_jsonl
from src.utils.json_schema_utils import is_valid_json


def _safe_parse_reference(ref_text: str) -> Optional[Any]:
    ok, obj = is_valid_json(ref_text or "")
    if not ok:
        return None
    return obj


def _type_compatible(candidate: Any, expected: Any) -> bool:
    # JSON parsing yields Python types; allow int/float interchange to reduce
    # spurious "wrong_type" when the teacher uses e.g. 22 vs 22.0.
    if isinstance(expected, bool) or isinstance(candidate, bool):
        return isinstance(candidate, bool) and isinstance(expected, bool)
    if isinstance(expected, (int, float)) and isinstance(candidate, (int, float)):
        return True
    return type(candidate) is type(expected)


def _infer_expected_dict_schema(ref_obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(ref_obj, dict):
        return None
    return ref_obj


def _schema_compliance_details(candidate_obj: Any, ref_obj: Any) -> Tuple[bool, Dict[str, Any]]:
    if not isinstance(ref_obj, dict) or not isinstance(candidate_obj, dict):
        return False, {"reason": "non_object"}

    expected_keys = set(ref_obj.keys())
    candidate_keys = set(candidate_obj.keys())

    missing = sorted(list(expected_keys - candidate_keys))
    extra = sorted(list(candidate_keys - expected_keys))

    if missing or extra:
        # Keep details so we can populate the error taxonomy.
        return False, {"reason": "key_mismatch", "missing_keys": missing, "extra_fields": extra}

    wrong_types: List[str] = []
    for k in expected_keys:
        ev = ref_obj.get(k)
        cv = candidate_obj.get(k)
        if isinstance(ev, list):
            if not isinstance(cv, list):
                wrong_types.append(k)
                continue
            # If the reference list is empty, just check it stays a list.
            if not ev:
                continue
            # Otherwise check that each element is compatible with the reference element type.
            # (best-effort; we don't know the teacher's full validation rules)
            ref_el_types = {type(x) for x in ev}
            cand_el_types = {type(x) for x in cv}
            if not (cand_el_types <= ref_el_types or any(t in ref_el_types for t in cand_el_types)):
                # For numeric lists, allow int/float mismatch.
                if all(isinstance(x, (int, float)) for x in cv) and all(isinstance(x, (int, float)) for x in ev):
                    continue
                wrong_types.append(k)
        else:
            if not _type_compatible(cv, ev):
                wrong_types.append(k)

    if wrong_types:
        return False, {"reason": "wrong_types", "wrong_type_keys": wrong_types}

    return True, {}


def _exact_match(candidate_obj: Any, ref_obj: Any) -> bool:
    # Strict deep equality (list order matters).
    return candidate_obj == ref_obj


def _field_level_f1_extraction(candidate_obj: Dict[str, Any], ref_obj: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Best-effort field-level F1.
    - Scalars: exact match gives P=R=1 else P=R=0.
    - Lists: treat as sets for F1.
    """
    if not ref_obj:
        return 0.0, {}

    f1s: Dict[str, float] = {}
    for key, ref_val in ref_obj.items():
        cand_val = candidate_obj.get(key, None)
        if isinstance(ref_val, list):
            ref_set = set(ref_val)
            cand_set = set(cand_val) if isinstance(cand_val, list) else set()
            if not ref_set and not cand_set:
                precision = recall = 1.0
            elif not cand_set and ref_set:
                precision = 0.0
                recall = 0.0
            else:
                inter = len(ref_set & cand_set)
                precision = inter / len(cand_set) if cand_set else 0.0
                recall = inter / len(ref_set) if ref_set else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            f1s[key] = f1
        else:
            if cand_val is not None and cand_val == ref_val:
                f1s[key] = 1.0
            else:
                f1s[key] = 0.0

    macro_f1 = sum(f1s.values()) / len(f1s) if f1s else 0.0
    return macro_f1, f1s


def _is_extraction_task_from_reference(ref_obj: Any) -> bool:
    """
    We use the reference JSON shape to decide whether this row is an
    "extraction" task.

    The current prediction artifacts may not include `task_type`, so relying on
    `r["task_type"]` can yield 0 extraction prompts.
    """
    if not isinstance(ref_obj, dict):
        return False
    # Our teacher/generated JSON extraction examples include the key `entities`.
    if "entities" in ref_obj:
        return True
    # Fallback for other extraction-like shapes.
    return ("location" in ref_obj and "date" in ref_obj)


def main() -> None:
    # Allow custom checkpoint names for ablations (env override).
    stage2_ckpt = os.getenv("STAGE2_CKPT_LABEL", "ckpt2_stage2")
    checkpoints = os.getenv("JSON_EVAL_CKPTS", "ckpt0_base,ckpt1_stage1," + stage2_ckpt).split(",")

    tables_dir = Path("artifacts/tables")
    metrics_dir = Path("artifacts/metrics")
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows_summary: List[Dict[str, Any]] = []

    # For JSON error taxonomy, we store counts per checkpoint and category.
    taxonomy_rows: List[Dict[str, Any]] = []

    for ckpt in checkpoints:
        pred_path = f"artifacts/predictions/{ckpt}_json_eval_outputs.jsonl"
        rows = read_jsonl(pred_path)
        total = len(rows)

        json_validity = 0
        schema_compliance = 0
        exact_match = 0

        extraction_f1s: List[float] = []
        extraction_count = 0

        taxonomy = Counter()

        for r in rows:
            pred_text = r.get("prediction", "") or ""
            ref_text = r.get("reference", "") or ""

            ref_obj = _safe_parse_reference(ref_text)
            if ref_obj is None:
                # Reference should always be valid, but guard anyway.
                taxonomy["reference_parse_failed"] += 1
                continue

            ok, cand_obj = is_valid_json(pred_text)
            if not ok:
                json_validity += 0
                # Heuristic taxonomy: missing closing brace suggests truncation.
                if "{" in pred_text and "}" not in pred_text:
                    taxonomy["truncated_output"] += 1
                else:
                    taxonomy["invalid_json"] += 1
                continue

            json_validity += 1
            if not isinstance(cand_obj, dict):
                taxonomy["not_object"] += 1
                continue

            compliant, details = _schema_compliance_details(cand_obj, ref_obj)
            if compliant:
                schema_compliance += 1
            else:
                reason = details.get("reason", "schema_compliance_failed")
                taxonomy[reason] += 1

            if _exact_match(cand_obj, ref_obj):
                exact_match += 1
            else:
                # Only taxonomize "exact mismatch" after schema compliance checks,
                # so we can distinguish formatting vs semantic correctness.
                taxonomy["exact_mismatch"] += 1

            # Field-level F1 for extraction tasks only.
            if _is_extraction_task_from_reference(ref_obj):
                extraction_count += 1
                f1, _ = _field_level_f1_extraction(cand_obj, ref_obj)
                extraction_f1s.append(f1)

        json_validity_rate = json_validity / total if total else 0.0
        # Assignment: schema compliance is among valid JSON outputs only.
        schema_compliance_rate = schema_compliance / json_validity if json_validity else 0.0
        # Assignment: exact-match accuracy is over all outputs.
        exact_match_rate = exact_match / total if total else 0.0
        extraction_f1_avg = sum(extraction_f1s) / len(extraction_f1s) if extraction_f1s else 0.0

        metrics = {
            "checkpoint": ckpt,
            "total": total,
            "json_validity_rate": json_validity_rate,
            "schema_compliance_rate": schema_compliance_rate,
            "exact_match_rate": exact_match_rate,
            "extraction_field_level_f1_avg": extraction_f1_avg,
            "extraction_prompts": extraction_count,
            "error_taxonomy_counts": dict(taxonomy),
        }

        write_jsonl(str(metrics_dir / f"json_auto_metrics_{ckpt}.json"), [metrics])

        rows_summary.append(
            {
                "checkpoint": ckpt,
                "json_validity_rate": json_validity_rate,
                "schema_compliance_rate": schema_compliance_rate,
                "exact_match_rate": exact_match_rate,
                "extraction_field_level_f1_avg": extraction_f1_avg,
                "extraction_prompts": extraction_count,
            }
        )

        for cat, cnt in taxonomy.items():
            taxonomy_rows.append({"checkpoint": ckpt, "category": cat, "count": cnt})

        print(
            f"[json-auto] {ckpt}: validity={json_validity_rate:.3f} "
            f"schema={schema_compliance_rate:.3f} exact={exact_match_rate:.3f} "
            f"extraction_F1={extraction_f1_avg:.3f} ({extraction_count} prompts)"
        )

    # Write summary CSVs for easy inclusion in the blog/report.
    # 1) per-checkpoint metrics table
    t1_path = tables_dir / "json_metrics_by_checkpoint.csv"
    import csv

    with t1_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "json_validity_rate",
                "schema_compliance_rate",
                "exact_match_rate",
                "extraction_field_level_f1_avg",
                "extraction_prompts",
            ],
        )
        w.writeheader()
        for row in rows_summary:
            w.writerow(row)

    # 2) error taxonomy table
    t2_path = tables_dir / "json_error_taxonomy_by_checkpoint.csv"
    with t2_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint", "category", "count"])
        w.writeheader()
        for row in taxonomy_rows:
            w.writerow(row)

    print(f"[json-auto] wrote {t1_path} and {t2_path}")


if __name__ == "__main__":
    main()
