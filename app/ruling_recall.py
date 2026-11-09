# -*- coding: utf-8 -*-
"""裁定库召回：父块路由 / 卡片缩池 / 全库向量 三路加权 RRF。"""
import json
from collections import defaultdict

import numpy as np

from . import config
from .embed_client import embed_one
from .rule_recall import match_parents

_s = {}
GENERIC_PARENTS = {"效果的原则 - 效果的处理", "发动条件", "效果的处理"}


def _load():
    if _s:
        return _s
    lib = {}
    for line in (config.DATA / "qa_library.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            lib[o["id"]] = o
    ids = [json.loads(l)["id"] for l in (config.DATA / "qa_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    vec = np.load(config.DATA / "qa_vec_f16.npy").astype("float32")
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    parent_idx = json.loads((config.DATA / "ruling_parent_index.json").read_text(encoding="utf-8"))
    card2qa = defaultdict(list)
    for qid, rec in lib.items():
        for cid in rec.get("cards") or []:
            card2qa[str(cid)].append(qid)
    _s.update({"lib": lib, "ids": ids, "vec": vec, "parent_idx": parent_idx,
               "pos": {q: i for i, q in enumerate(ids)}, "card2qa": card2qa})
    return _s


def _depth():
    return int(config.get("retrieval.lane_depth", 50))


def _lane_parent(query, s, hits):
    if not hits:
        return []
    spec = [k for k in hits if k not in GENERIC_PARENTS]
    spec.sort(key=lambda k: -hits[k])
    keep = set(spec[:int(config.get("retrieval.rule_top_parents", 4))]) | {k for k in hits if k in GENERIC_PARENTS}
    cnt = defaultdict(int)
    for name in keep:
        if name in GENERIC_PARENTS:
            continue
        for qid in s["parent_idx"].get(name, ()):
            cnt[qid] += 1
    cand = [q for q in cnt if q in s["pos"]]
    if not cand:
        return []
    rows = [s["pos"][q] for q in cand]
    sim = s["vec"][rows] @ embed_one(query)
    order = sorted(range(len(cand)), key=lambda i: (-cnt[cand[i]], -sim[i]))
    return [cand[i] for i in order[:_depth()]]


def _lane_card(query, cids, s):
    pool = []
    for c in cids:
        pool.extend(s["card2qa"].get(str(c), ()))
    pool = [q for q in dict.fromkeys(pool) if q in s["pos"]]
    if not pool or len(pool) > 400:
        return []
    rows = [s["pos"][q] for q in pool]
    sim = s["vec"][rows] @ embed_one(query)
    order = np.argsort(sim)[::-1]
    return [pool[int(i)] for i in order[:_depth()]]


def _lane_vector(query, s):
    th = float(config.get("retrieval.vector_min_similarity", 0.55))
    sim = s["vec"] @ embed_one(query)
    order = np.argsort(sim)[::-1][:_depth()]
    return [s["ids"][int(i)] for i in order if sim[int(i)] >= th]


def recall_rulings(query, cids, hits, top=1):
    """返回 (picked, lanes_info, detail)"""
    s = _load()
    k = float(config.get("retrieval.rrf_k", 60))
    lanes = [
        ("parent", float(config.get("retrieval.w_parent", 0.7)), _lane_parent(query, s, hits)),
        ("card", float(config.get("retrieval.w_card", 1.0)), _lane_card(query, cids, s)),
        ("vector", float(config.get("retrieval.w_vector", 1.0)), _lane_vector(query, s)),
    ]
    score, detail = defaultdict(float), defaultdict(list)
    for name, w, ranked in lanes:
        for rank, qid in enumerate(ranked, 1):
            score[qid] += w / (k + rank)
            detail[qid].append([name, rank])
    info = [[n, len(r)] for n, _w, r in lanes]
    if not score:
        return None, info, {}
    best = sorted(score, key=lambda q: -score[q])[0]
    rec = s["lib"].get(best)
    if not rec:
        return None, info, {}
    return {"id": best, "score": round(score[best], 5), "lanes": detail[best], "record": rec}, info, detail
