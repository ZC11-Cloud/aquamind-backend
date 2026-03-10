from typing import List, Dict, Any, AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import SecretStr

class AIService:
    def __init__(self, api_key: str, model_name: str = "qwen-plus"):
        """
        初始化AI服务
        
        Args:
            api_key: DashScope API密钥
            model_name: 使用的模型名称
        """
        self.chat_model = ChatTongyi(model=model_name, api_key=api_key) # type: ignore

    async def generate_response(self, messages: List[Dict[str, str]], system_prompt: str = "your are a helpful assistant") -> str:
        """
        生成AI回复
        
        Args:
            messages: 消息历史记录，格式为[{"role": "user", "content": "消息内容"}, ...]
            system_prompt: 系统提示词
            
        Returns:
            AI生成的回复内容
        """
        # 构建消息列表
        langchain_messages = []
        
        # 添加系统提示词
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))
        
        # 转换消息历史记录
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        # 调用AI生成回复
        response = self.chat_model.invoke(langchain_messages)
        
        content = response.content
        if isinstance(content, list):
            content = str(content)
        return content

    async def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "your are a helpful assistant",
    ) -> AsyncGenerator[str, None]:
        """
        流式生成AI回复，按 token/chunk 逐步 yield 内容。

        Args:
            messages: 消息历史记录，格式为[{"role": "user", "content": "消息内容"}, ...]
            system_prompt: 系统提示词

        Yields:
            每个生成片段的字符串内容（delta）
        """
        langchain_messages = []

        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))

        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        async for chunk in self.chat_model.astream(langchain_messages):
            content = chunk.content
            if content is None:
                continue
            if isinstance(content, list):
                content = str(content)
            if isinstance(content, str) and content:
                yield content

    async def generate_short_title(self, user_question: str) -> str:
        """
        根据用户问题生成一句简短的中文对话标题。

        Args:
            user_question: 用户的首条问题内容

        Returns:
            生成的标题，不超过 15 个字
        """
        system_prompt = (
            "根据用户问题生成一句简短的中文对话标题，不超过 15 个字，不要引号、不要标点结尾。"
        )
        messages = [{"role": "user", "content": user_question}]
        result = await self.generate_response(messages, system_prompt)
        return (result or "").strip()[:255]


# 工厂函数，用于创建AIService实例
def create_ai_service(api_key: str, model_name: str = "qwen-plus") -> AIService:
    return AIService(api_key=api_key, model_name=model_name)