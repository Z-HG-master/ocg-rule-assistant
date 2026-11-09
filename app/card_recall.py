# -*- coding: utf-8 -*-
"""卡片库召回：别名精确匹配 + 系列匹配。数据来自 data/card_runtime.json。"""
import json
import re
from collections import defaultdict

from . import config

_s = {}

BANNED_KEYS = {"无效"}                      # 与规则常用词冲突，封禁
CONTEXT_ONLY = {"融合": ("魔法卡融合", "融合魔法卡", "融合卡")}


def _load():
    if _s:
        return _s
    rt = json.loads((config.DATA / "card_runtime.json").read_text(encoding="utf-8"))
    idx = json.loads((config.DATA / "card_index_v30.json").read_text(encoding="utf-8"))
    _s.update({"keys": rt["keys"], "meta": rt["meta"],
               "archetypes": idx.get("archetypes", {})})
    return _s


def _norm(text):
    return re.sub(r"[\s·・\-－—☆★\"'“”「」『』()（）]", "", str(text or "")).lower()


def recall_cards(query, top=6):
    """返回 [(cid, name, types, desc)]"""
    s = _load()
    t = _norm(query)
    ids = []
    for key, cids in s["keys"].items():
        if len(key) < 2 or len(key) > 12 or key not in t:
            continue
        if key in BANNED_KEYS:
            continue
        ctx = CONTEXT_ONLY.get(key)
        if ctx and not any(c in t for c in ctx):
            continue
        ids.extend(int(c) for c in cids)
    for name, cids in s["archetypes"].items():
        if len(name) >= 3 and _norm(name) in t:
            ids.extend(int(c) for c in cids[:3])
    out, seen = [], set()
    for c in ids:
        if c in seen:
            continue
        seen.add(c)
        m = s["meta"].get(str(c))
        if m:
            out.append((c, m.get("name"), m.get("types"), m.get("desc") or ""))
        if len(out) >= top:
            break
    return out


def card_line(item):
    cid, name, types, desc = item
    body = (desc or "").replace("\n", " ").strip()
    return "[卡片 %s] %s｜%s｜%s" % (cid, name, types or "", body[:400])
