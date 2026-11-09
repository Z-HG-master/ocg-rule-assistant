# ocg-rules-rag

游戏王 OCG 规则助手 · 检索层（v1.1）：**卡片库 + 规则库 + 裁定库** 三库召回，裁定召回为「父块路由 / 卡片缩池 / 全库向量」三路加权 RRF 融合。自带一个本地测试台，分「检索召回测试」与「AI 测试」两个入口。

## 快速开始

```bash
pip install -r requirements.txt

# 复制配置并填写你自己的 API 地址与模型
cp config.example.json config.json

# 启动测试台
python run.py
# → http://127.0.0.1:7871
```

数据库已随仓库携带，拉下来即可运行，无需自行构建索引。

## 依赖的两个 API

两个都用 OpenAI 兼容接口，地址、模型名、key 全部在 `config.json` 中配置。

### 1. Embedding

```json
"embedding": {
  "base_url": "https://api.your-provider.com/v1",
  "model": "text-embedding-qwen3-embedding-4b",
  "api_key": "",
  "dimension": 2560
}
```

- 模型为 **qwen3-embedding-4b**（`text-embedding-qwen3-embedding-4b`），输出 **2560 维**
- 任何 OpenAI 兼容的 embedding 服务都可以，填自己的 `base_url` 与 `api_key`
- **维度必须一致**（库里的向量就是 2560 维），换模型前先确认维度

### 2. Chat

```json
"chat": {
  "base_url": "https://api.your-provider.com/v1",
  "model": "your-chat-model",
  "api_key": "",
  "temperature": 0.3,
  "max_tokens": 2048
}
```

「AI 测试」标签页会先用检索层拼出资料块，再把资料与问题一起发给这个接口。

## 检索参数

全部在 `config.json` 的 `retrieval` 段：

| 键 | 含义 |
| --- | --- |
| `rrf_k` | RRF 平滑常数，默认 60 |
| `w_parent` | 父块路由这一路的权重（默认 0.7） |
| `w_card` | 卡片缩池这一路的权重（默认 1.0） |
| `w_vector` | 全库向量这一路的权重（默认 1.0） |
| `lane_depth` | 每一路取前多少名参与融合 |
| `vector_min_similarity` | 向量路的最低相似度门槛 |
| `rule_top_blocks` | 注入的规则块数量 |
| `card_top` | 注入的卡片数量 |

RRF 得分：`Σ 路权重 / (k + 该路排名)`。

## 三个库各怎么召回

### 卡片库

- 别名表精确匹配（官方名、中文译名、系列名、昵称，共 44,280 个键）
- 系列名命中时按字段展开
- 通用词屏蔽：`无效` 作为卡名与规则用语冲突，已封禁；`融合` 只在出现「魔法卡融合 / 融合魔法卡 / 融合卡」时认作卡

### 规则库

- **串联父块**：247 条路径，按路径前缀命中（如「效果的处理 - 个别效果处理的例子 - 场所的移动 - 除外」）
- **语义父块**：861 个概念标签，问题向量与父块向量相似度 ≥ 0.70 时参与
- 命中父块后取直接挂靠的规则块，按命中权重排序

### 裁定库

11,527 条官方中文 FAQ，每条都已标注规则父块，反向索引为 `data/ruling_parent_index.json`。

三路并行，各出排名，再用加权 RRF 融合：

1. **父块路由**：问题命中的规则父块 → 取这些父块下的裁定 → 按命中父块数与相似度排序
2. **卡片缩池**：问题涉及的卡 → 该卡关联的裁定 → 按相似度排序
3. **全库向量**：11,527 条问句向量直接检索

## 目录

```
app/
  config.py          配置加载
  embed_client.py    Embedding API（qwen3-embedding-4b，2560 维）
  chat_client.py     对话 API
  card_recall.py     卡片库召回
  rule_recall.py     规则库召回（串联父块 + 语义父块）
  ruling_recall.py   裁定库召回（三路加权 RRF）
  retrieve.py        三库合并入口
  webui.py           测试台（两个标签页）
data/
  card_runtime.json          别名键 → cid，cid → 卡文
  card_index_v30.json        卡片索引（14,378 张 / 440 系列 / 84,010 别名）
  rule_nodes_v31.json        规则父块表（247 串联 / 260 前缀 / 64 同义 / 861 语义）
  rulebook_chunks_v32.jsonl  规则块（1,060 条，带串联前缀正文）
  rule_parents_vec_v32.npy   规则父块向量
  rule_parents_meta_v32.jsonl 规则父块元信息
  qa_library.jsonl           裁定库 11,527 条
  qa_meta.jsonl              裁定元信息
  qa_vec_f16.npy             裁定问句向量（float16，56MB）
  ruling_parent_index.json   父块 → 裁定反向索引
```

## 注意

- `qa_vec_f16.npy` 是 float16 存储（原始 float32 为 118MB，超出 GitHub 单文件限制），加载时自动转回 float32。抽样验证最近邻与 float32 完全一致。
- `config.json` 在 `.gitignore` 中，不会提交；仓库里只有 `config.example.json`。
