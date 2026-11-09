# -*- coding: utf-8 -*-
"""规则库召回：父块路由 + 词项加权段落打分。

流程：
  1) 用户问句 → 派生检索串（正则词表扩展，无模型调用）
  2) 检索串 → 加权词表（原词 / 概念族成员 / 中文长词切片 / 泛用词降权）
  3) 引号锚点「」『』单独提权
  4) 父块匹配（结构串联父块 + 语义父块）得到候选块
  5) 逐块打分：正文命中 × 长度加成，串联标题命中 × 1.5
  6) 总分 = 父块加成 + 块打分；无父块命中时退回全库词项扫描
"""
import json
import re

import numpy as np

from . import config
from .embed_client import embed_one

SEM_THRESHOLD = 0.70
_s = {}

VOCAB_RULES = [
    (r"(发动|発動)", "发动 発動 发动条件"),
    (r"(连锁|チェーン|chain)", "连锁 チェーン 组成连锁 连锁处理"),
    (r"(处理|適用|适用|结算)", "处理 適用 解决 效果处理"),
    (r"(手卡|手牌|手札|hand)", "手卡 手牌 加入手卡"),
    (r"(回到|返回|放回|弹回)", "回到 返回 回到卡组 回到手卡"),
    (r"(墓地|送墓)", "墓地 送去墓地"),
    (r"(除外|banish)", "除外 除外状态"),
    (r"(破坏|破壊|destroy)", "破坏 破坏效果"),
    (r"(攻击|攻撃|attack)", "攻击 战斗 伤害步骤"),
    (r"(次数|回数|[一二三四五六七八九十\d]+次)", "次数 一次 一回合一次"),
    (r"(不受.{0,12}(效果|効果).{0,6}影响)", "不受效果影响 效果的对象"),
    (r"(无效|無効|negate|康|挡|反制|counter)", "无效 效果的无效 发动的无效 召唤无效"),
    (r"(伤害步骤结束|ダメージステップ終了)", "伤害步骤结束时 伤害计算"),
    (r"(取对象|対象|对象)", "取对象 效果的对象"),
    (r"(cost|コスト|解放|支付)", "cost 解放 支付"),
]


def _norm(t):
    return re.sub(r"[\s·・\-－—☆★\"'“”「」『』]", "", str(t or "")).lower()


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
    concepts = json.loads((config.DATA / "concept_groups.json").read_text(encoding="utf-8"))
    _s.update({"nodes": nodes, "blocks": blocks, "meta": meta, "vec": vec, "concepts": concepts})
    return _s


def derive_query(user_query):
    q = str(user_query or "")
    adds = []
    for pat, words in VOCAB_RULES:
        if re.search(pat, q, re.I):
            adds.append(words)
    return " ".join([q] + adds), adds


def build_terms(user_query, concepts):
    w = concepts["weights"]
    gen = set(concepts["generic_terms"]["terms"])
    gweight = float(concepts["generic_terms"]["weight"])
    query_text, fired = derive_query(user_query)
    terms = {}

    def add(term, weight):
        key = _norm(term)
        if len(key) < 2:
            return
        terms[key] = max(terms.get(key, 0.0), float(weight))

    for seg in re.split(r"[\s，,。.!！?？;；、:：()（）\[\]【】「」『』《》/]+", query_text):
        seg = seg.strip()
        if len(seg) < 2:
            continue
        base = gweight if seg in gen else w["user_term"]
        add(seg, base)
        if len(seg) > 6 and re.search(r"[\u3400-\u9fff]", seg):
            for size in range(2, min(w["ngram_max"], len(seg)) + 1):
                for i in range(len(seg) - size + 1):
                    add(seg[i:i + size], base * w["ngram_factor"] * size)

    for fam in concepts["families"]:
        if any(re.search(re.escape(t), query_text, re.I) for t in fam["trigger"]):
            for m in fam["members"]:
                add(m, w["user_term"] * fam["factor"] * w["family_factor"])
                if m in gen:
                    add(m, gweight)
    return terms, fired


def extract_anchors(user_query):
    out = []
    for m in re.findall(r"「([^」]{2,40})」|『([^』]{2,40})』", str(user_query or "")):
        s = (m[0] or m[1]).strip()
        if s:
            out.append(_norm(s))
    return out


def match_parents(user_query):
    s = _load()
    q = _norm(user_query)
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
    v = embed_one(user_query)
    sim = s["vec"] @ v
    for i in np.argsort(sim)[::-1][:24]:
        if sim[i] < SEM_THRESHOLD:
            break
        m = s["meta"][int(i)]
        if m["kind"] != "semantic":
            continue
        hits[m["name"]] = max(hits.get(m["name"], 0.0), float(sim[i]))
    return hits


def score_block(block, terms, anchors, concepts):
    w = concepts["weights"]
    text = _norm(block.get("text") or "")
    title = _norm(block.get("own") or "")
    chain = _norm(block.get("chain") or "")
    score = 0.0
    strong = 0
    for term, weight in terms.items():
        if term in text:
            boost = min(3.0, max(0.5, len(term) / 4.0))
            score += weight * boost
            if title and term in title:
                score += weight * boost * w["title_boost"]
            if len(term) >= 4 and weight >= w["user_term"]:
                strong += 1
        elif term in chain:
            score += weight * 0.25
    hit_anchors = [a for a in anchors if a and a in text]
    for a in hit_anchors:
        score += w["anchor_base"] + min(w["anchor_len_cap"], len(a) / 2.0)
    if len(hit_anchors) >= 2:
        score += len(hit_anchors) * w["anchor_multi_bonus"]
    if not strong and score < w["min_score"]:
        return 0.0
    return score


def recall_rules(user_query, top_blocks=3, return_debug=False):
    s = _load()
    concepts = s["concepts"]
    w = concepts["weights"]
    hits = match_parents(user_query)
    terms, fired = build_terms(user_query, concepts)
    anchors = extract_anchors(user_query)
    full_scan = False
    if not hits and not anchors:
        if len(terms) < 6:
            dbg0 = {"terms": len(terms), "fired": fired, "anchors": 0, "candidates": 0, "scores": []}
            return ([], [], dbg0) if return_debug else ([], [])
        full_scan = True

    cand = {}
    for name, hv in hits.items():
        if name in s["nodes"]["chains"]:
            for bid in s["nodes"]["chains"].get(name, []):
                cand[bid] = max(cand.get(bid, 0.0), w["parent_struct_bonus"] * hv)
        for bid in s["nodes"]["semantic"]["blocks"].get(name, []):
            cand[bid] = max(cand.get(bid, 0.0), w["parent_semantic_bonus"] * hv)
    if (anchors or full_scan) and not cand:
        cand = {bid: 0.0 for bid in s["blocks"]}

    scored = []
    for bid, bonus in cand.items():
        b = s["blocks"].get(bid)
        if not b:
            continue
        sc = score_block(b, terms, anchors, concepts)
        if sc <= 0 and not bonus:
            continue
        scored.append((bonus + sc, bid, sc, bonus))
    scored.sort(key=lambda x: -x[0])
    top = scored[:top_blocks]
    blocks = [s["blocks"][bid] for _t, bid, _s, _b in top]
    parents = sorted(hits, key=lambda k: -hits[k])[:6]
    dbg = {"terms": len(terms), "fired": fired, "anchors": len(anchors), "candidates": len(cand),
           "scores": [(bid, round(sc, 2), round(bn, 2)) for _t, bid, sc, bn in top]}
    return (blocks, parents, dbg) if return_debug else (blocks, parents)
