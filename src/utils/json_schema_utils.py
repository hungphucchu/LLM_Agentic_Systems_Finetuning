import json
from typing import Any, Dict, Tuple


def is_valid_json(text: str) -> Tuple[bool, Any]:
    try:
        obj = json.loads(text)
        return True, obj
    except Exception:
        return False, None


def has_required_schema_keys(obj: Dict[str, Any], required_keys: Dict[str, type]) -> bool:
    for key, typ in required_keys.items():
        if key not in obj:
            return False
        if not isinstance(obj[key], typ):
            return False
    return True
