# -*- coding: utf-8 -*-
"""配置加载：仓库根目录的 config.json 优先，缺失时用 config.example.json。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

_cache = {}


def load():
    if _cache:
        return _cache
    for name in ("config.json", "config.example.json"):
        p = ROOT / name
        if p.exists():
            _cache.update(json.loads(p.read_text(encoding="utf-8")))
            _cache["_source"] = name
            return _cache
    raise SystemExit("找不到 config.json 或 config.example.json")


def get(path, default=None):
    node = load()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
