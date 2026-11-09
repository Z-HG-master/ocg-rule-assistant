# -*- coding: utf-8 -*-
"""对话 API：OpenAI 兼容接口，模型与地址都在 config.json 里改。"""
import json
import urllib.request

from . import config


def chat(messages, max_tokens=None, temperature=None):
    base = config.get("chat.base_url").rstrip("/")
    body = {
        "model": config.get("chat.model"),
        "messages": messages,
        "temperature": float(temperature if temperature is not None else config.get("chat.temperature", 0.3)),
        "max_tokens": int(max_tokens if max_tokens is not None else config.get("chat.max_tokens", 2048)),
    }
    key = config.get("chat.api_key") or ""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    return (d["choices"][0]["message"].get("content") or "").strip()
