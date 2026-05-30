import json
import logging
import asyncio
import contextlib
from fastapi import APIRouter, Depends, File, HTTPException, Request, status, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import os
from dotenv import load_dotenv

from src.dependencies import get_session, get_current_user
from src.models.user import User
from src.models.qa import QaConversation
from src.schemas.qa import (
    QaConversationCreate,
    QaConversationResponse,
    QaConversationListResponse,
    QaConversationTitleResponse,
    QaSuggestionsResponse,
    QaMessageCreate,
    QaMessageResponse,
    QaMessageListResponse,
    QaAttachmentUploadResponse,
)
from src.service.qa_service import QaService
from src.service.ai_service import create_ai_service
from src.service.knowledge_service import create_knowledge_service
from src.service.rag_service import create_rag_service
from src.service.agent_service import create_agent_service
from src.service.yolo_service import get_yolo_service
from src.settings import DASHSCOPE_API_KEY, QA_ATTACHMENT_UPLOAD_DIR
from src.service.knowledge_upload_service import save_upload_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/qa', tags=["qa"])

if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY is not configured")
ai_service = create_ai_service(api_key=DASHSCOPE_API_KEY)
knowledge_service = create_knowledge_service()
rag_service = create_rag_service(ai_service=ai_service, knowledge_service=knowledge_service)
_yolo_service = None
try:
    _yolo_service = get_yolo_service()
except FileNotFoundError:
    pass
agent_service = create_agent_service(
    ai_service=ai_service,
    knowledge_service=knowledge_service,
    yolo_service=_yolo_service,
)


def _get_qa_service(session):
    return QaService(
        session, ai_service,
        rag_service=rag_service,
        agent_service=agent_service,
    )


@router.post("/attachments", response_model=QaAttachmentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_chat_attachment(
    file: UploadFile = File(..., description="聊天附件（PDF/TXT/MD/DOCX）"),
    current_user: User = Depends(get_current_user),
):
    _ = current_user
    stored = await save_upload_file(file, QA_ATTACHMENT_UPLOAD_DIR)
    return QaAttachmentUploadResponse(
        file_id=stored.file_id,
        original_filename=stored.original_filename,
        file_ext=stored.file_ext,
        size=stored.size,
    )
@router.post("/conversations", response_model=QaConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation_data: QaConversationCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """创建新的会话"""
    qa_service = _get_qa_service(session)
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
    """发送消息到会话；请求体中 use_rag=true 时将基于知识库检索回答。"""
    qa_service = _get_qa_service(session)
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
        reasoning_content=message.reasoning_content,
        image_url=message.image_url,
        attachments=message.attachments,
        citations=message.citations,
        create_time=message.create_time,
    )


@router.post("/conversations/{conversation_id}/messages/stream")
async def send_message_stream(
    request: Request,
    conversation_id: int,
    message_data: QaMessageCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """流式发送消息到会话，返回 SSE 流；支持 use_rag/use_image/image_base64，Agent 自动或按开关调用工具。"""
    # 调试：请求体中的开关与图片
    logger.info(
        "[DEBUG] stream 请求体: use_rag=%s, use_image=%s, image_base64 有值=%s, content_len=%d",
        getattr(message_data, "use_rag", None),
        getattr(message_data, "use_image", None),
        bool(getattr(message_data, "image_base64", None)),
        len(getattr(message_data, "content", "") or ""),
    )
    qa_service = _get_qa_service(session)

    async def event_stream():
        cancel_event = asyncio.Event()
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=16)

        async def producer():
            try:
                async for chunk in qa_service.send_message_stream(
                    conversation_id,
                    current_user.id,
                    message_data,
                    cancel_event=cancel_event,
                ):
                    if cancel_event.is_set():
                        break
                    if isinstance(chunk, dict):
                        payload = json.dumps(chunk, ensure_ascii=False)
                    else:
                        payload = json.dumps(
                            {"type": "chunk", "content": chunk},
                            ensure_ascii=False,
                        )
                    await queue.put(f"data: {payload}\n\n")
            except asyncio.CancelledError:
                cancel_event.set()
                raise
            except ValueError as e:
                await queue.put(
                    f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"
                )
            except Exception as e:
                logger.exception("流式消息异常: %s", e)
                await queue.put(
                    f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"
                )
            finally:
                await queue.put(None)

        producer_task = asyncio.create_task(producer())
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    logger.info(
                        "检测到客户端断开，取消流式生成, conversation_id=%s",
                        conversation_id,
                    )
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if item is None:
                    break
                yield item
        finally:
            cancel_event.set()
            try:
                await asyncio.wait_for(producer_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "流式任务优雅退出超时，执行强制取消, conversation_id=%s",
                    conversation_id,
                )
                producer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await producer_task
            except asyncio.CancelledError:
                with contextlib.suppress(asyncio.CancelledError):
                    await producer_task

    # 校验失败抛出 ValueError；其他异常统一 yield error 事件供前端展示
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations", response_model=QaConversationListResponse)
async def get_conversations(
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """获取用户的会话列表"""
    qa_service = _get_qa_service(session)
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
    qa_service = _get_qa_service(session)
    messages, total = await qa_service.get_messages(conversation_id, current_user.id, skip, limit)
    
    return QaMessageListResponse(
        messages=[QaMessageResponse.model_validate(m) for m in messages],
        total=total
    )


@router.post(
    "/conversations/{conversation_id}/generate-title",
    response_model=QaConversationTitleResponse,
)
async def generate_conversation_title(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """根据首轮对话内容生成并更新会话标题，仅当消息数为 2 且标题为「新对话」时执行。"""
    qa_service = _get_qa_service(session)
    title = await qa_service.generate_conversation_title(conversation_id, current_user.id)
    if title is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不满足生成条件（会话不存在、消息数非 2、或标题已非「新对话」）",
        )
    return QaConversationTitleResponse(title=title)


@router.get(
    "/conversations/{conversation_id}/suggestions",
    response_model=QaSuggestionsResponse,
)
async def get_suggestions(
    conversation_id: int,
    message_id: int | None = None,
    limit: int = 3,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """获取与当前上下文相关的追问建议。"""
    qa_service = _get_qa_service(session)
    try:
        suggestions = await qa_service.get_suggestions(
            conversation_id=conversation_id,
            user_id=current_user.id,
            message_id=message_id,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    return QaSuggestionsResponse(suggestions=suggestions)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """删除会话及其所有消息"""
    qa_service = _get_qa_service(session)
    success = await qa_service.delete_conversation(conversation_id, current_user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在或不属于该用户"
        )
    
    # 返回204 No Content表示成功删除
    return None
