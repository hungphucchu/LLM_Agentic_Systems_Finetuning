import json
import re
from typing import Any, Dict, List, Optional, Tuple


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

    # Remove common reasoning tags (Phi <think>, Qwen3 `</think>`, etc.).
    # Paired blocks first; then keep only text after the last closing tag, since
    # Qwen often emits long CoT before the final JSON answer.
    _reason_name = r"(?:think|thinking|redacted_reasoning)"
    t = re.sub(
        rf"<\s*{_reason_name}\s*>.*?<\s*/\s*{_reason_name}\s*>",
        "",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    _close_reason = re.compile(rf"<\s*/\s*{_reason_name}\s*>\s*", re.IGNORECASE)
    chunks = _close_reason.split(t)
    if len(chunks) > 1:
        t = chunks[-1].strip()
    t = re.sub(rf"</?\s*{_reason_name}\s*>", "", t, flags=re.IGNORECASE)

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


def assistant_message_text(resp: Any) -> str:
    """
    Best-effort assistant string from an OpenAI-compatible chat.completions response.
    Handles empty ``content`` when the server puts text in ``reasoning_content`` or
    multimodal-style ``content`` lists.
    """
    try:
        ch = resp.choices[0]
        msg = getattr(ch, "message", None)
        if msg is None:
            return ""

        content = getattr(msg, "content", None)
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text") or ""))
                elif isinstance(part, str):
                    parts.append(part)
            joined = "".join(parts).strip()
            if joined:
                return joined

        if isinstance(content, str) and content.strip():
            return content.strip()

        for attr in ("reasoning_content", "refusal"):
            val = getattr(msg, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()

        return (content or "").strip() if isinstance(content, str) else ""
    except (IndexError, AttributeError, TypeError):
        return ""


def parse_llm_json_dict(text: Optional[str]) -> Dict[str, Any]:
    """
    Parse a single JSON object from LLM output using tolerant normalization
    (markdown fences, leading commentary, etc.).
    """
    ok, obj = is_valid_json(text or "")
    if ok and isinstance(obj, dict):
        return obj
    preview = repr((text or "")[:500])
    raise ValueError(f"expected JSON object, ok={ok} preview={preview}")


def has_required_schema_keys(obj: Dict[str, Any], required_keys: Dict[str, type]) -> bool:
    for key, typ in required_keys.items():
        if key not in obj:
            return False
        if not isinstance(obj[key], typ):
            return False
    return True
