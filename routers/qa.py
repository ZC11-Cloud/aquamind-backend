from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from dependencies import get_session, get_current_user
from models.user import User
from models.qa import QaConversation
from schemas.qa import (
    QaConversationCreate,
    QaConversationResponse,
    QaConversationListResponse,
    QaMessageCreate,
    QaMessageResponse,
    QaMessageListResponse
)
from service.qa_service import QaService

router = APIRouter(prefix='/qa', tags=["qa"])


@router.post("/conversations", response_model=QaConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation_data: QaConversationCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """创建新的会话"""
    qa_service = QaService(session)
    conversation = await qa_service.create_conversation(current_user.id, conversation_data)
    # 在返回前确保所有属性已加载
    await session.refresh(conversation)
    return QaConversationResponse(
        id=conversation.id,
        user_id=current_user.id,
        title=conversation_data.title,
        create_time=conversation.create_time,
        update_time=conversation.update_time
    )


@router.post("/conversations/{conversation_id}/messages", response_model=QaMessageResponse, status_code=status.HTTP_201_CREATED)
async def send_message(
    conversation_id: int,
    message_data: QaMessageCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """发送消息到会话"""
    qa_service = QaService(session)
    message = await qa_service.send_message(conversation_id, current_user.id, message_data)
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在或不属于该用户"
        )
    await session.refresh(message)
    return QaMessageResponse(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        create_time=message.create_time
    )


@router.get("/conversations", response_model=QaConversationListResponse)
async def get_conversations(
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """获取用户的会话列表"""
    qa_service = QaService(session)
    conversations, total = await qa_service.get_conversations(current_user.id, skip, limit)

    return QaConversationListResponse(
        conversations=[QaConversationResponse.model_validate(c) for c in conversations],
        total=total
    )


@router.get("/conversations/{conversation_id}/messages", response_model=QaMessageListResponse)
async def get_messages(
    conversation_id: int,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """获取会话的聊天记录"""
    qa_service = QaService(session)
    messages, total = await qa_service.get_messages(conversation_id, current_user.id, skip, limit)
    
    return QaMessageListResponse(
        messages=[QaMessageResponse.model_validate(m) for m in messages],
        total=total
    )