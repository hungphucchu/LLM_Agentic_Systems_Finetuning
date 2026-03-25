import json
import re
from typing import Any, Dict, Tuple


def is_valid_json(text: str) -> Tuple[bool, Any]:
    try:
        candidate = _normalize_json_text(text)
        obj = json.loads(candidate)
        return True, obj
    except Exception:
        return False, None


def _normalize_json_text(text: str) -> str:
    """
    Make JSON parsing more tolerant to common LLM formatting, e.g.:
    - ```json ... ```
    - leading/trailing commentary
    - JSON object embedded inside a longer message
    """
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return t

    # Remove common reasoning tags that some models include (e.g., Phi/Qwen/others).
    # We remove both block and inline forms so JSON extraction can work.
    t = re.sub(
        r"<\s*(?:think|thinking)\s*>.*?<\s*/\s*(?:think|thinking)\s*>",
        "",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    t = re.sub(r"</?\s*(?:think|thinking)\s*>", "", t, flags=re.IGNORECASE)

    # Remove ```json / ``` fenced blocks (best-effort).
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    # If the model included commentary, try to extract the first JSON container.
    # Prefer objects, then arrays.
    obj_start = t.find("{")
    obj_end = t.rfind("}")
    arr_start = t.find("[")
    arr_end = t.rfind("]")

    # Choose the earliest valid start among { or [
    starts = [(obj_start, "obj"), (arr_start, "arr")]
    starts = [(i, kind) for i, kind in starts if i != -1]
    if not starts:
        return t
    starts.sort(key=lambda x: x[0])
    first_kind = starts[0][1]

    if first_kind == "obj" and obj_end != -1 and obj_end > obj_start:
        candidate = t[obj_start : obj_end + 1]
    elif first_kind == "arr" and arr_end != -1 and arr_end > arr_start:
        candidate = t[arr_start : arr_end + 1]
    else:
        return t

    # Best-effort JSON repairs for common LLM formatting errors.
    # 1) Remove trailing commas: { "a": 1, } -> { "a": 1 }
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    return candidate


def has_required_schema_keys(obj: Dict[str, Any], required_keys: Dict[str, type]) -> bool:
    for key, typ in required_keys.items():
        if key not in obj:
            return False
        if not isinstance(obj[key], typ):
            return False
    return True
