
import requests
import json
import time

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"


def propose_corrections(suspicious_values, allowed_values):

    prompt = f"""
You are a data cleaning assistant.

Suggest corrections for suspicious categorical values.

Rules:
1. Only suggest values from the allowed values list.
2. Never invent new categories.
3. Preserve valid meanings such as "unknown".
4. If uncertain, use null.
5. Do not modify any dataset.
6. Return valid JSON only.

Allowed values:
{json.dumps(allowed_values, ensure_ascii=False)}

Suspicious values and frequencies:
{json.dumps(suspicious_values, ensure_ascii=False)}

Return this JSON structure:
{{
    "proposals": [
        {{
            "original_value": "example",
            "suggested_value": "example",
            "reason": "short explanation",
            "confidence": 0.0
        }}
    ]
}}
"""

    start = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 1000
            }
        },
        timeout=300
    )

    response.raise_for_status()

    result = response.json()
    elapsed = time.perf_counter() - start

    content = result["message"]["content"]

    return {
        "response": json.loads(content),
        "runtime_seconds": elapsed,
        "prompt_tokens": result.get("prompt_eval_count"),
        "completion_tokens": result.get("eval_count"),
        "total_duration_ns": result.get("total_duration"),
        "load_duration_ns": result.get("load_duration")
    }
