from datetime import datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Query, Depends
from loguru import logger
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.core.security import get_current_active_user
from app.crud.chat_session import insert_chat, get_chat_list
from app.crud.knowledge_base import get_knowledge_bases
from app.db.session import SessionDep
from app.models import UserTable, ChatSessionTable
from app.schemas.chat_session import ChatSession
from app.schemas.knowledge_base import KnowledgeBase
from llm.core.model import load_glm4_flash


class ChatRequest(BaseModel):
    message: str
    knowledge_base_name: str
    history: str


router = APIRouter()


@router.get('/knowledge_bases', response_model=list[KnowledgeBase])
async def read_knowledge_bases(
        session: SessionDep,
        offset: int = 0,
        limit: Annotated[int, Query(le=20)] = 20,
):
    return get_knowledge_bases(session, offset, limit)


@router.get('/chats', response_model=list[ChatSession])
async def get_chats(session: SessionDep, knowledge_base_name: str):
    # TODO 这里是测试用，后面记得改
    return get_chat_list(session, 'admin@example.com', knowledge_base_name)


@router.patch('/{knowledge_base_name}')
async def add_new_chat(session: SessionDep, knowledge_base_name: str):
    now_time = datetime.now()

    new_chat = ChatSessionTable(
        chat_history=uuid4(),
        knowledge_base_name=knowledge_base_name,
        user_email='admin@example.com',  # TODO 这里是测试用，后面记得改
        create_time=now_time,
        update_time=now_time
    )
    insert_chat(session, new_chat)
    return new_chat.chat_history


@router.post('/chat')
async def chat(
        request: ChatRequest,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    llm = load_glm4_flash()

    logger.info(f'{current_user.username}: {request.message}')

    async def generate():
        full_response = ''
        async for chunk in llm.astream(request.message):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
        logger.info(f'AI: {full_response}')

    return StreamingResponse(
        generate(),
        media_type='text/event-stream'
    )
