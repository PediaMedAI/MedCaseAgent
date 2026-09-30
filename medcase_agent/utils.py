"""Shared API and formatting helpers for preprocessing and report generation."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path


def load_environment(env_file=None):
    """Load an explicit dotenv file, preserving variables already in the shell."""
    from dotenv import load_dotenv

    path = Path(env_file) if env_file else Path.cwd() / ".env"
    if env_file and not path.is_file():
        raise ValueError(f"Environment file does not exist: {path}")
    if path.is_file():
        load_dotenv(path, override=False)


def get_openai_client():
    """Create an OpenAI-compatible client from environment configuration."""
    from openai import OpenAI

    load_environment()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key in {"your-api-key", "your_api_key"}:
        raise ValueError("Set OPENAI_API_KEY in your environment or .env file.")
    options = {
        "api_key": key,
        "timeout": float(os.getenv("OPENAI_TIMEOUT", "180")),
        "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "2")),
    }
    if os.getenv("OPENAI_BASE_URL"):
        options["base_url"] = os.environ["OPENAI_BASE_URL"]
    return OpenAI(**options)


def extract_json_from_text(content):
    """Read a JSON object, allowing a surrounding Markdown code fence."""
    if not isinstance(content, str):
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # raw_decode handles braces in quoted strings and surrounding prose.
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                parsed, _ = decoder.raw_decode(text[match.start():])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None
    return parsed if isinstance(parsed, dict) else None


def encode_image(image_path):
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def finalize_prompt(prompt_text):
    return re.sub(r"\n{3,}", "\n\n", "\n".join(
        line.strip() for line in prompt_text.splitlines()
    )).strip()


def generate_llm_response(client, model, messages, stream=False, **kwargs):
    """Normalize Chat Completions responses and streamed native tool calls."""
    max_tokens = os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "12000")
    token_parameter = os.getenv("OPENAI_MAX_TOKENS_PARAM", "max_tokens")
    if token_parameter not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("OPENAI_MAX_TOKENS_PARAM must be max_tokens or max_completion_tokens")
    supplied_tokens = kwargs.pop("max_tokens", None)
    if "max_completion_tokens" not in kwargs:
        kwargs[token_parameter] = supplied_tokens if supplied_tokens is not None else int(max_tokens)
    if os.getenv("OPENAI_SEND_TEMPERATURE", "true").lower() in {"false", "0", "no"}:
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
    if stream:
        kwargs.setdefault("stream_options", {"include_usage": True})
    response = client.chat.completions.create(
        model=model, messages=messages, stream=stream, **kwargs
    )
    if not stream:
        if not response.choices:
            raise RuntimeError("The model returned no completion choices.")
        message = response.choices[0].message
        calls = [{
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        } for call in (getattr(message, "tool_calls", None) or [])]
        return {"content": message.content or "", "tool_calls": calls or None,
                "reasoning_content": "", "usage": getattr(response, "usage", None)}

    content, calls, usage = [], {}, None
    for chunk in response:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content.append(delta.content)
            print(delta.content, end="", flush=True)
        for call in (getattr(delta, "tool_calls", None) or []):
            item = calls.setdefault(call.index, {
                "id": "", "type": "function", "function": {"name": "", "arguments": ""},
            })
            if getattr(call, "id", None):
                item["id"] = call.id
            if getattr(call, "function", None):
                if getattr(call.function, "name", None):
                    item["function"]["name"] += call.function.name
                item["function"]["arguments"] += getattr(call.function, "arguments", "") or ""
    return {"content": "".join(content), "tool_calls": [calls[i] for i in sorted(calls)] or None,
            "reasoning_content": "", "usage": usage}


def format_clinical_section(section_data):
    if not section_data:
        return "Information not provided in the case record."
    if isinstance(section_data, str):
        return section_data.strip()
    if isinstance(section_data, list):
        return "\n".join("- " + (
            ", ".join(f"{k}: {v}" for k, v in item.items())
            if isinstance(item, dict) else str(item).strip()
        ) for item in section_data)
    if isinstance(section_data, dict):
        return "\n".join(f"- **{key.capitalize()}**: {value}" for key, value in section_data.items())
    return str(section_data).strip()
