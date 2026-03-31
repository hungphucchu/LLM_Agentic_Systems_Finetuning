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


def _balanced_brace_objects(s: str) -> List[str]:
    """Top-level balanced `{...}` spans in order of closing (outer segments first)."""
    out: List[str] = []
    stack: List[int] = []
    for i, c in enumerate(s):
        if c == "{":
            stack.append(i)
        elif c == "}" and stack:
            start = stack.pop()
            if not stack:
                out.append(s[start : i + 1])
    return out


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

    # Qwen/vLLM: unclosed reasoning prefix (no `</think>` yet) — start at first `{`.
    if re.match(rf"^<\s*{_reason_name}\s*>", t, re.IGNORECASE) and "{" in t:
        cut = t.find("{")
        if cut != -1:
            t = t[cut:].lstrip()

    # Remove ```json / ``` fenced blocks (best-effort).
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    # Prefer a JSON *object* (judge schema). Reasoning often contains literal arrays like [1,2,3]
    # before the real `{ ... }`; do not take the earliest `[`..`]` slice.
    for seg in reversed(_balanced_brace_objects(t)):
        candidate = re.sub(r",\s*([}\]])", r"\1", seg)
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return candidate

    obj_start = t.find("{")
    obj_end = t.rfind("}")
    arr_start = t.find("[")
    arr_end = t.rfind("]")

    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidate = t[obj_start : obj_end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return candidate

    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidate = t[arr_start : arr_end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return candidate

    return t


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
    if ok and isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        return obj[0]
    preview = repr((text or "")[:500])
    got = f"{type(obj).__name__}" if ok else "n/a"
    raise ValueError(f"expected JSON object, ok={ok} parsed_type={got} preview={preview}")


def has_required_schema_keys(obj: Dict[str, Any], required_keys: Dict[str, type]) -> bool:
    for key, typ in required_keys.items():
        if key not in obj:
            return False
        if not isinstance(obj[key], typ):
            return False
    return True
