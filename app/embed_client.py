# -*- coding: utf-8 -*-
"""Embedding 走 API：默认调用 LM Studio 的 qwen3-embedding-4b（2560 维）。"""
import json
import urllib.request

import numpy as np

from . import config


def _post(url, payload, api_key=""):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def embed(texts):
    """返回归一化后的 float32 向量矩阵 (n, dim)。"""
    if isinstance(texts, str):
        texts = [texts]
    base = config.get("embedding.base_url").rstrip("/")
    model = config.get("embedding.model")
    key = config.get("embedding.api_key") or ""
    batch = int(config.get("embedding.batch", 32))
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        d = _post(base + "/embeddings", {"model": model, "input": chunk}, key)
        out.extend(item["embedding"] for item in d["data"])
    M = np.asarray(out, dtype="float32")
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


def embed_one(text):
    return embed([text])[0]
