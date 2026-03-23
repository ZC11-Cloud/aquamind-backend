"""
将 LangChain / DashScope 多模态接口返回的 message.content 规范为纯文本。

多模态 generation 常返回 list[dict]，如 [{'text': '你好'}]，若用 str(list) 会得到
"[{'text': '你好'}]" 导致前端乱码；流式时每 chunk 也可能为同类结构。
"""
from __future__ import annotations

from typing import Any


def normalize_message_content(content: Any) -> str:
    """
    将 AIMessage / chunk 的 content 转为可展示的字符串。

    - str：原样返回
    - list：逐项提取 dict 的 text（或 type=text 时的 text），其余 str() 兜底拼接
    - None：空串
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
                    continue
                # 少数结构
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                    continue
                inner = item.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
                elif inner is not None:
                    parts.append(normalize_message_content(inner))
                else:
                    # 避免把整段 dict repr 进回复
                    parts.append("")
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)
