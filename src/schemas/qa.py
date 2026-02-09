from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class QaMessageCreate(BaseModel):
    """创建消息的请求模型"""
    content: str
    use_rag: bool = False  # 是否基于知识库检索回答（RAG）


class QaMessageResponse(BaseModel):
    """消息响应模型"""
    id: int
    conversation_id: int
    role: str
    content: str
    create_time: datetime

    class Config:
        from_attributes = True
        use_enum_values = True


class QaConversationCreate(BaseModel):
    """创建会话的请求模型"""
    title: Optional[str] = None


class QaConversationResponse(BaseModel):
    """会话响应模型"""
    id: int
    user_id: int
    title: Optional[str]
    create_time: datetime
    update_time: datetime

    class Config:
        from_attributes = True
        use_enum_values = True


class QaConversationListResponse(BaseModel):
    """会话列表响应模型"""
    conversations: List[QaConversationResponse]
    total: int


class QaMessageListResponse(BaseModel):
    """消息列表响应模型"""
    messages: List[QaMessageResponse]
    total: int