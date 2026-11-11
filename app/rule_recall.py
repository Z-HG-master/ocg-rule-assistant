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

# 简中 ↔ 官方缩写：匹配前把问句里的简中术语折算成规则库口径
SYNONYM_TO_SHORT = [
    ("同调素材", "S素材"), ("同调怪兽", "S怪兽"), ("同调召唤", "S召唤"),
    ("同调", "S"), ("超量素材", "X素材"), ("超量怪兽", "X怪兽"),
    ("超量召唤", "X召唤"), ("超量", "X"), ("额外卡组", "EX卡组"),
    ("灵摆效果", "P效果"), ("灵摆区域", "P区域"), ("灵摆怪兽", "P怪兽"),
    ("灵摆召唤", "P召唤"), ("灵摆", "P"),
]

# 展示用：把缩写补成「S（同调）」这种括号同义写法
SHORT_TO_LONG = {
    "S素材": "S素材（同调素材）", "S怪兽": "S怪兽（同调怪兽）", "S召唤": "S召唤（同调召唤）",
    "X素材": "X素材（超量素材）", "X怪兽": "X怪兽（超量怪兽）", "X召唤": "X召唤（超量召唤）",
    "P效果": "P效果（灵摆效果）", "P区域": "P区域（灵摆区域）", "P怪兽": "P怪兽（灵摆怪兽）",
    "P召唤": "P召唤（灵摆召唤）", "EX卡组": "EX卡组（额外卡组）",
}


def to_short(text):
    """把问句里的简中术语折算为规则库缩写，便于父块匹配。"""
    out = str(text or "")
    for a, b in SYNONYM_TO_SHORT:
        out = out.replace(a, b)
    return out


def expand_display(text):
    """展示时给缩写补括号同义词。

    三种情况跳过，避免出现「P区域（灵摆区域）」被再包一层：
      1. 缩写前面紧跟「（」——已经在括号里
      2. 缩写后面紧跟「（」——已经带了注记
      3. 缩写后面紧跟「）」——括号内的缩写
    """
    out = str(text or "")
    for a, b in SHORT_TO_LONG.items():
        pattern = r"(?<![A-Za-z（(])" + a + r"(?![A-Za-z）)])"
        out = re.sub(pattern, b, out)
    return out

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
    q_short = _norm(to_short(user_query))       # 简中问句折算成 X/S/P/EX 后再匹配
    hits = {}
    for name in s["nodes"]["chains"]:
        for part in name.split(" - "):
            n = _norm(part)
            if len(n) >= 2 and (n in q or n in q_short):
                hits[name] = max(hits.get(name, 0.0), 1.0)
    for alias, target in s["nodes"]["alias"].items():
        n = _norm(alias)
        if len(n) >= 2 and (n in q or n in q_short):
            hits[target] = max(hits.get(target, 0.0), 1.0)
    v = embed_one(to_short(user_query))      # 向量也走缩写口径，与规则库用语对齐
    sim = s["vec"] @ v
    for i in np.argsort(sim)[::-1][:24]:
        if sim[i] < SEM_THRESHOLD:
            break
        m = s["meta"][int(i)]
        if m["kind"] != "semantic":
            continue
        hits[m["name"]] = max(hits.get(m["name"], 0.0), float(sim[i]))
    return hits


def question_tags(user_query, concepts):
    """问句命中的标签（用于块内细分排序）。"""
    rules = concepts.get("tag_rules") or {}
    trig = rules.get("tags") or {}
    q = str(user_query or "")
    out = set()
    for tag, words in trig.items():
        for w in words:
            try:
                if re.search(w, q):
                    out.add(tag)
                    break
            except re.error:
                if w in q:
                    out.add(tag)
                    break
    return out


def score_block(block, terms, anchors, concepts, qtags=None):
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
    rules = concepts.get("tag_rules") or {}
    btags = (rules.get("block_tags") or {}).get(block.get("bid"))
    if qtags and btags:
        overlap = len(qtags & set(btags))
        if overlap:
            score += overlap * float(rules.get("weight", 2.0))
    if not strong and score < w["min_score"]:
        return 0.0
    return score


def _lane_parent(s, hits, terms, anchors, concepts, qtags=None):
    """路 A：父块路由（命中的父块下挂的块）。"""
    w = concepts["weights"]
    cand = {}
    for name, hv in hits.items():
        if name in s["nodes"]["chains"]:
            for bid in s["nodes"]["chains"].get(name, []):
                cand[bid] = max(cand.get(bid, 0.0), w["parent_struct_bonus"] * hv)
        for bid in s["nodes"]["semantic"]["blocks"].get(name, []):
            cand[bid] = max(cand.get(bid, 0.0), w["parent_semantic_bonus"] * hv)
    if not cand:
        return []
    scored = []
    for bid, bonus in cand.items():
        b = s["blocks"].get(bid)
        if not b:
            continue
        sc = score_block(b, terms, anchors, concepts, qtags)
        scored.append((bonus + sc, bid))
    scored.sort(key=lambda x: -x[0])
    return [bid for _t, bid in scored]


def _lane_terms(s, terms, anchors, concepts, qtags=None):
    """路 B：全库逐块词项打分（对齐参考实现：不依赖父块命中）。"""
    scored = []
    for bid, b in s["blocks"].items():
        sc = score_block(b, terms, anchors, concepts, qtags)
        if sc > 0:
            scored.append((sc, bid))
    scored.sort(key=lambda x: -x[0])
    return [bid for _t, bid in scored]


def recall_rules(user_query, top_blocks=3, return_debug=False):
    s = _load()
    concepts = s["concepts"]
    w = concepts["weights"]
    hits = match_parents(user_query)
    terms, fired = build_terms(user_query, concepts)
    anchors = extract_anchors(user_query)
    if not terms:
        dbg0 = {"terms": 0, "fired": fired, "anchors": 0, "lanes": (0, 0), "scores": []}
        return ([], [], dbg0) if return_debug else ([], [])

    qtags = question_tags(user_query, concepts)
    lane_a = _lane_parent(s, hits, terms, anchors, concepts, qtags)
    lane_b = _lane_terms(s, terms, anchors, concepts, qtags)
    k = float(w.get("rrf_k", 60))
    wa = float(w.get("w_parent_lane", 0.8))
    wb = float(w.get("w_term_lane", 1.0))
    score = {}
    rank_a, rank_b = {}, {}
    for i, bid in enumerate(lane_a, 1):
        score[bid] = score.get(bid, 0.0) + wa / (k + i)
        rank_a[bid] = i
    for i, bid in enumerate(lane_b, 1):
        score[bid] = score.get(bid, 0.0) + wb / (k + i)
        rank_b[bid] = i
    top = sorted(score, key=lambda b: -score[b])[:top_blocks]

    blocks = []
    for bid in top:
        b = dict(s["blocks"][bid])
        b["display"] = expand_display(b.get("display") or "")
        blocks.append(b)
    parents = sorted(hits, key=lambda k2: -hits[k2])[:6]
    dbg = {"terms": len(terms), "fired": fired, "anchors": len(anchors),
           "lanes": (len(lane_a), len(lane_b)),
           "scores": [(bid, round(score[bid], 5), rank_a.get(bid), rank_b.get(bid)) for bid in top]}
    return (blocks, parents, dbg) if return_debug else (blocks, parents)
