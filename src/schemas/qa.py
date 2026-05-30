from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class QaAttachment(BaseModel):
    file_id: str
    original_filename: str
    file_ext: Optional[str] = None
    size: Optional[int] = None


class QaAttachmentUploadResponse(QaAttachment):
    pass


class KnowledgeCitation(BaseModel):
    """知识库引用，供前端 Sources 组件溯源。"""
    key: int
    source_id: str
    filename: str
    snippet: str


class QaMessageCreate(BaseModel):
    """创建消息的请求模型"""
    content: str
    use_rag: bool = False  # 是否基于知识库检索回答（RAG）
    use_image: bool = False  # 是否使用图像识别（需同时提供 image_base64）
    image_base64: Optional[str] = None  # 可选，base64 编码的图片，与 use_image 配合
    # 可选：指定模型（如 qwen3.5-plus、qwen3-max、qwen3-vl-plus 等），不传则走默认
    attachments: Optional[List[QaAttachment]] = None
    model_name: Optional[str] = None
    # 深度思考参数（OpenAI 兼容扩展参数）
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None
    preserve_thinking: Optional[bool] = None


class QaMessageResponse(BaseModel):
    """消息响应模型"""
    id: int
    conversation_id: int
    role: str
    content: str
    reasoning_content: Optional[str] = None
    image_url: Optional[str] = None
    attachments: Optional[List[QaAttachment]] = None
    citations: Optional[List[KnowledgeCitation]] = None
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


class QaConversationTitleResponse(BaseModel):
    """生成标题响应模型"""
    title: str


class QaSuggestionsResponse(BaseModel):
    """追问建议响应模型"""
    suggestions: List[str]
