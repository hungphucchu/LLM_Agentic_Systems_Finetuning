"""OpenAI-compatible client setup for teacher and judge API calls."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from src.utils.json_schema_utils import assistant_message_text, parse_llm_json_dict

DEFAULT_BASE_URL = "http://10.246.100.230/v1"
DEFAULT_JUDGE_MODEL = "Llama-3.1-70B-Instruct-custom"


def openai_endpoint() -> tuple[str, str]:
    load_dotenv()
    base_url = os.getenv("BASE_URL") or os.getenv("UTSA_BASE_URL") or DEFAULT_BASE_URL
    base_url = base_url.strip().replace("Links to an external site.", "").strip()
    api_key = os.getenv("API_KEY") or os.getenv("UTSA_API_KEY") or "EMPTY"
    if isinstance(api_key, str):
        api_key = api_key.strip()
    return base_url, api_key


def sync_teacher_model_from_utsa() -> None:
    if os.getenv("TEACHER_MODEL") is None and os.getenv("UTSA_MODEL") is not None:
        os.environ["TEACHER_MODEL"] = os.environ["UTSA_MODEL"]


def build_openai_client_for_judge(*, log_prefix: str) -> OpenAI:
    base_url, api_key = openai_endpoint()
    if api_key == "EMPTY":
        print(f"[{log_prefix}] Warning: API_KEY/UTSA_API_KEY is not set; judge calls may fail.")
    model = os.getenv("JUDGE_MODEL", os.getenv("UTSA_MODEL", DEFAULT_JUDGE_MODEL))
    print(f"[{log_prefix}] Using base_url={base_url} model={model}")
    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    client._judge_model = model  # type: ignore[attr-defined]
    return client


def build_openai_client_for_teacher() -> OpenAI:
    load_dotenv()
    sync_teacher_model_from_utsa()
    base_url, api_key = openai_endpoint()
    timeout_seconds = float(os.getenv("TEACHER_TIMEOUT_SECONDS", "120"))
    if api_key == "EMPTY":
        print("[teacher-gen] Warning: API_KEY/UTSA_API_KEY is not set. Using API_KEY=EMPTY.")
    effective = os.getenv("TEACHER_MODEL", "")
    print(f"[teacher-gen] Using base_url={base_url} model={effective}")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)


def call_judge_json_completion(
    client: OpenAI,
    *,
    system_message: str,
    user_prompt: str,
    log_prefix: str,
    failure_label: str = "Judge",
    max_retries: int = 3,
) -> Dict[str, Any]:
    model = getattr(client, "_judge_model")
    max_tokens = int(os.getenv("JUDGE_MAX_TOKENS", "4096"))
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            text = assistant_message_text(resp)
            if not text:
                fr = getattr(resp.choices[0], "finish_reason", None)
                raise ValueError(f"empty judge response (finish_reason={fr})")
            return parse_llm_json_dict(text)
        except (APITimeoutError, RateLimitError, APIError, json.JSONDecodeError, ValueError) as e:
            last_err = e
            print(
                f"[{log_prefix}] retry {attempt + 1}/{max_retries + 1} after error: "
                f"{type(e).__name__}: {e}"
            )
            continue
        except Exception as e:  # pragma: no cover
            last_err = e
            break
    raise RuntimeError(f"{failure_label} failed after {max_retries + 1} attempts: {last_err}")
