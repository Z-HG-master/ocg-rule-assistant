# -*- coding: utf-8 -*-
"""规则库召回：串联父块（结构）+ 语义父块，父块命中后取直接挂靠的子块。"""
import json
import re

import numpy as np

from . import config
from .embed_client import embed_one

_s = {}
THRESHOLD = 0.70


def _norm(text):
    return re.sub(r"[\s·・\-－—☆★\"'“”「」『』]", "", str(text or "")).lower()


def _load():
    if _s:
        return _s
    nodes = json.loads((config.DATA / "rule_nodes_v31.json").read_text(encoding="utf-8"))
    blocks = {}
    for line in (config.DATA / "rulebook_chunks_v32.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            b = json.loads(line)
            blocks[b["bid"]] = b
    meta = [json.loads(l) for l in (config.DATA / "rule_parents_meta_v32.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    vec = np.load(config.DATA / "rule_parents_vec_v32.npy").astype("float32")
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    _s.update({"nodes": nodes, "blocks": blocks, "meta": meta, "vec": vec})
    return _s


def match_parents(query):
    """返回 {父块名: 权重}；结构父块正则优先，语义父块用向量兜底。"""
    s = _load()
    q = _norm(query)
    hits = {}
    for name in s["nodes"]["chains"]:
        for part in name.split(" - "):
            n = _norm(part)
            if len(n) >= 3 and n in q:
                hits[name] = max(hits.get(name, 0.0), 1.0)
    for alias, target in s["nodes"]["alias"].items():
        n = _norm(alias)
        if len(n) >= 3 and n in q:
            hits[target] = max(hits.get(target, 0.0), 1.0)
    v = embed_one(query)
    sim = s["vec"] @ v
    for i in np.argsort(sim)[::-1][:24]:
        if sim[i] < THRESHOLD:
            break
        m = s["meta"][int(i)]
        if m["kind"] != "semantic":
            continue
        hits[m["name"]] = max(hits.get(m["name"], 0.0), float(sim[i]))
    return hits


def recall_rules(query, top_blocks=3):
    s = _load()
    hits = match_parents(query)
    if not hits:
        return [], []
    score = {}
    for name, w in hits.items():
        for b in s["nodes"]["chains"].get(name, []):
            score[b] = score.get(b, 0) + 2 * w
        for b in s["nodes"]["semantic"]["blocks"].get(name, []):
            score[b] = score.get(b, 0) + w
    ranked = sorted(score, key=lambda b: -score[b])[:top_blocks]
    blocks = [s["blocks"][b] for b in ranked if b in s["blocks"]]
    parents = sorted(hits, key=lambda k: -hits[k])[:6]
    return blocks, parents
