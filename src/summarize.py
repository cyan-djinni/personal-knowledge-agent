"""
使用 Google 官方 API 对抓取的内容进行总结
"""

import json
import os
import time
from google import genai
from src.fetchers.base import RawItem

_PROMPT_BASE = """
## 内容处理规则（按类型）

### 播客 / YouTube 字幕（source_type: youtube_transcript 或 follow_builders）
每集写一段 150-300 字的精华提炼：
- 第一句：「核心观点」——这集最重要的一个结论是什么？
- 介绍主讲人的身份背景（姓名、公司/职位）
- 提炼 2-3 个反直觉、具体、或有实操价值的洞见，避免泛泛而谈
- 引用一句原文中最有力的话（原文语言）
- 写法：像聪明的朋友在给你做口头总结，直接切入内容，不写「本期节目」「主持人问道」之类的废话

### Twitter/X 推文（source_type: follow_builders 且来自 @handle）
每个人写 2-4 句话：
- 先介绍此人身份：全名 + 职位/公司（如「Replit CEO Amjad Masad」）
- 只写实质性内容：原创观点、产品动态、技术讨论、行业判断
- 跳过：日常碎碎念、转发、「活动很棒！」类内容
- 如果有大胆预测或反主流观点，优先提
- 如果该人无实质内容，写「本期无实质更新」

### 文章 / RSS（source_type: rss 或 email）
每篇写：
- **核心论点**：一句话说清楚这篇在讲什么
- **关键要点**：如果正文内容充实（超过 500 字），列出 5-10 条 bullet points，每条一句话，聚焦具体观点、数据、结论或反直觉见解；内容较短则列 2-3 条即可，不要凑数
- **值不值得深读**：一句话说明理由
- 附原文链接（原生保留，不要翻译链接）

### 暂无更新的来源（title == "暂无更新"）
在「来源详情」中仍然列出该来源，写一句：「今日无新内容」。不要跳过，不要省略。
**将所有「今日无新内容」的来源统一列在「来源详情」最末尾，不要穿插在有内容的来源之间。**

## 强制要求
- 每一条有实质内容的条目都必须附原始链接；没有链接的不要写进摘要
- 暂无更新的来源：只写来源名称 + 「今日无新内容」，不需要链接
- 不要编造任何内容，只基于 JSON 中给出的数据
- 保持原文语言特征：如果是德文/英文来源，保留其原生的关键专有名词或标题
- 格式要适合手机阅读：段落之间空行，重点用加粗
- 不使用破折号（——除外）开头的列表
"""

_PROMPTS = {
    "zh": f"""你是一个专业的信息与个人知识助手。我会给你一段 JSON，包含从多个来源抓取的最新内容。
请全程使用中文输出，技术词汇（AI、LLM、API、RAG、token、agent 等）及德语/英语专有名词保留原文，人名和产品名保留英文或原语言。

## 输出结构

### 今日要点
跨所有来源，按重要性排出 5 条最值得关注的内容，每条一段话（不是一句话），说清楚为什么重要。

### 来源详情
按来源逐一展开，每个来源单独一节，标题写来源名称。

{_PROMPT_BASE}""",

    "en": f"""You are a personal knowledge and digest assistant. I will give you a JSON containing the latest content fetched from multiple sources.
Output entirely in English.

## Output Structure

### Today's Highlights
Pick the 5 most important items across all sources, ranked by importance. Each item gets a full paragraph (not a single sentence) explaining why it matters.

### Source Breakdown
Go through each source one by one, with its name as a section heading.

{_PROMPT_BASE}""",

    "bilingual": f"""You are a personal knowledge digest assistant. I will give you a JSON containing the latest content fetched from multiple sources.
Output in bilingual format: Chinese and English interleaved paragraph by paragraph.

Rules:
- Section headings in both languages: e.g. "## 今日要点 / Today's Highlights"
- For each item: write the Chinese paragraph first, then the English paragraph directly below (blank line between), then move to the next item
- Technical terms and proper nouns stay in their original language

## Output Structure

### 今日要点 / Today's Highlights
5 most important items, each gets a paragraph in both languages.

### 来源详情 / Source Breakdown
Each source as a separate section.

{_PROMPT_BASE}""",
}


def prepare_digest(items: list[RawItem]) -> dict:
    """将 RawItem 列表整理成结构化 JSON（不调用 LLM）。"""
    groups: dict[str, list[dict]] = {}
    for item in items:
        entry = {
            "source": item.source_name,
            "title": item.title,
            "content": item.content,
        }
        if item.link:
            entry["link"] = item.link
        if item.published:
            entry["published"] = item.published
        groups.setdefault(item.source_type, []).append(entry)
    return {"sources": groups, "total_items": len(items)}


# 将 model 改为官方提示的 "gemini-3.6-flash"
def summarize(
    items: list[RawItem],
    api_key: str | None = None,
    model: str = "gemini-3.6-flash",
    language: str = "zh",
) -> str:


    """调用 Google 官方 GenAI SDK 生成总结。"""
    if not items:
        no_content = {"zh": "暂无新内容。", "en": "No new content.", "bilingual": "暂无新内容。/ No new content."}
        return no_content.get(language, "暂无新内容。")

    raw_key = os.getenv("GEMINI_API_KEY") or api_key
    if not raw_key:
        raise ValueError("❌ 未配置 GEMINI_API_KEY！请检查 .env 文件。")

    clean_key = raw_key.strip().strip("'").strip('"')

    digest = prepare_digest(items)
    content = json.dumps(digest, ensure_ascii=False, indent=2)
    system_prompt = _PROMPTS.get(language, _PROMPTS["zh"])
    full_prompt = f"{system_prompt}\n\n以下是待处理的 JSON 数据:\n{content}"

    # 使用 Google 官方原生客户端
    client = genai.Client(api_key=clean_key)

    for attempt in range(3):
        try:
            print("🚀 正通过 Gemini 官方 SDK 请求生成每日知识摘要...")
            response = client.models.generate_content(
                model=model,
                contents=full_prompt,
            )
            return response.text
        except Exception as e:
            print(f"[Gemini 尝试 {attempt + 1}/3 失败]: {e}")
            if attempt < 2:
                time.sleep(5)
                continue
            raise e
