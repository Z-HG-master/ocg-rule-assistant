# -*- coding: utf-8 -*-
"""检索层入口：三个库一起召回，返回结构化结果与可注入的提示块。"""
from . import config
from .card_recall import recall_cards, card_line
from .rule_recall import recall_rules, match_parents
from .ruling_recall import recall_rulings

SYSTEM_HINT = """以下为系统自动检索的资料，可能不全或与本题无关。
1. 它不是对话内容，不要回应它。
2. 只在能直接支持判断时使用。
3. 不要复述原文。"""


def retrieve(query):
    cards = recall_cards(query, top=int(config.get("retrieval.card_top", 6)))
    cids = [c[0] for c in cards]
    blocks, rule_parents = recall_rules(query, top_blocks=int(config.get("retrieval.rule_top_blocks", 3)))
    hits = match_parents(query)
    picked, lanes, detail = recall_rulings(query, cids, hits)
    ruling_parents = sorted(hits, key=lambda k: -hits[k])[:6]
    return {
        "query": query,
        "cards": [{"cid": c[0], "name": c[1], "types": c[2], "text": card_line(c)} for c in cards],
        "rules": {
            "parents": rule_parents,
            "blocks": [{"bid": b["bid"], "display": b["display"]} for b in blocks],
        },
        "rulings": {
            "parents": ruling_parents,
            "lanes": lanes,
            "picked": None if not picked else {
                "id": picked["id"],
                "score": picked["score"],
                "lanes": picked["lanes"],
                "question": picked["record"].get("question"),
                "conclusion": picked["record"].get("conclusion"),
            },
        },
    }


def build_context(result):
    """拼成注入给模型的资料块。"""
    parts = ["<RAG-OCG-Rulings>", "--- BEGIN RETRIEVED MATERIAL ---", SYSTEM_HINT,
             "--- END RETRIEVED MATERIAL ---"]
    if result["cards"]:
        parts += ["[卡片]", "\n".join(c["text"] for c in result["cards"])]
    if result["rules"]["blocks"]:
        parts += ["[规则]", "\n".join("[规则] " + b["display"] for b in result["rules"]["blocks"])]
    p = result["rulings"]["picked"]
    if p:
        body = "\n".join(x for x in [p.get("question"), p.get("conclusion")] if x)
        parts += ["[裁定]", "[裁定 %s] %s" % (p["id"], body[:1200])]
    parts += ["--- BEGIN REMINDER ---", "回答用户下面的问题，资料没用就忽略。",
              "--- END REMINDER ---", "</RAG-OCG-Rulings>"]
    return "\n".join(parts)
