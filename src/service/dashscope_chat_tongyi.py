"""
LangChain ChatTongyi 扩展：按百炼要求为部分模型选择正确 API 端点。

- 纯文本模型（如 qwen-plus、qwen3-max）→ dashscope.Generation
  （POST .../text-generation/generation）
- 多模态模型（如 qwen3.5-plus、qwen3-vl-plus）→ dashscope.MultiModalConversation
  （POST .../multimodal-generation/generation）

官方说明见：InvalidParameter「url error」— 模型名称与 API 端点不匹配。
https://help.aliyun.com/zh/model-studio/error-code#error-url

上游 langchain_community 仅根据固定列表或名称中含「vl」选择多模态客户端，
未包含 qwen3.5-plus / qwen3.5-flash，会导致误走文本端点。
"""
from __future__ import annotations

from typing import Any, Dict

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init


EXPLICIT_MULTIMODAL_MODELS = {
    "qwen-audio-turbo",
    "qwen-audio-turbo-latest",
    "qwen-vl-plus",
    "qwen-vl-plus-latest",
    "qwen-vl-max",
    "qwen-vl-max-latest",
    # 文档明确：qwen3.5-plus / flash 需走 multimodal-generation
    "qwen3.5-plus",
    "qwen3.5-flash",
}


def is_multimodal_model(model_name: str | None) -> bool:
    """判断模型是否应走 DashScope 多模态端点。"""
    name = (model_name or "").strip().lower()
    return bool(name) and (name in EXPLICIT_MULTIMODAL_MODELS or "vl" in name)


class ChatTongyiDashScope(ChatTongyi):
    """与 ChatTongyi 行为一致，但将百炼要求走多模态端点的模型纳入 MultiModalConversation。"""

    @pre_init
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["dashscope_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "dashscope_api_key", "DASHSCOPE_API_KEY")
        )
        try:
            import dashscope
        except ImportError as e:
            raise ImportError(
                "Could not import dashscope python package. "
                "Please install it with `pip install dashscope --upgrade`."
            ) from e

        model_name = values.get("model_name") or values.get("model") or ""

        if is_multimodal_model(model_name):
            try:
                values["client"] = dashscope.MultiModalConversation
            except AttributeError as e:
                raise ValueError(
                    "`dashscope` has no `MultiModalConversation` attribute, upgrade "
                    "with `pip install dashscope --upgrade`."
                ) from e
        else:
            try:
                values["client"] = dashscope.Generation
            except AttributeError as e:
                raise ValueError(
                    "`dashscope` has no `Generation` attribute, upgrade dashscope."
                ) from e
        return values
