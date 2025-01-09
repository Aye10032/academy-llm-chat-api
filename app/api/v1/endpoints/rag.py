from datetime import datetime
from typing import Annotated, Any, Optional, Union
from uuid import uuid4
from enum import Enum
import json
import asyncio

from fastapi import APIRouter, Query, Depends
from langchain_community.chat_message_histories import SQLChatMessageHistory
from loguru import logger
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.core.security import get_current_active_user
from app.crud.chat_session import insert_chat, get_chat_list
from app.crud.knowledge_base import get_knowledge_bases
from app.crud.user import update_user
from app.db.session import SessionDep, engine
from app.models import UserTable, ChatSessionTable
from app.schemas.chat_session import ChatSession
from app.schemas.knowledge_base import KnowledgeBase
from app.schemas.user import UserUpdate
from llm.core.chat import rag_chat
from llm.core.model import load_glm4_flash, load_embedding, load_reranker
from llm.rag.retriever import base_retriever
from llm.rag.storage import get_vector_db, get_doc_db


class ChatRequest(BaseModel):
    message: str
    knowledge_base_name: str
    history: str


class ChatEventType(Enum):
    STATUS = "status"
    DOCS = "docs"
    ANSWER = "answer"


class SSEMessage(BaseModel):
    event: ChatEventType
    data: Union[str, list[dict], dict]

    def to_sse(self) -> str:
        """转换为 SSE 格式的消息"""
        return f"event: {self.event.value}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


router = APIRouter()


@router.get(
    '/knowledge_bases',
    description='返回知识库列表',
    response_model=list[KnowledgeBase]
)
async def read_knowledge_bases(
        session: SessionDep,
        offset: int = 0,
        limit: Annotated[int, Query(le=20)] = 20,
):
    return get_knowledge_bases(session, offset, limit)


@router.get('/chats', response_model=list[ChatSession])
async def get_chats(
        session: SessionDep,
        knowledge_base_name: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
):
    """返回用户在对应知识库下的对话列表

    Args:
        session: 数据库连接会话
        knowledge_base_name:
        current_user:

    Returns:

    """
    return get_chat_list(session, str(current_user.email), knowledge_base_name)


@router.patch('/{knowledge_base_name}', description='在对应知识库下新建对话')
async def add_new_chat(
        session: SessionDep,
        knowledge_base_name: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
):
    now_time = datetime.now()

    new_chat = ChatSessionTable(
        chat_history=str(uuid4()),
        knowledge_base_name=knowledge_base_name,
        user_email=str(current_user.email),
        create_time=now_time,
        update_time=now_time
    )
    insert_chat(session, new_chat)
    return new_chat.chat_history


@router.get('/chat/{chat_history}')
async def load_chat(
        chat_history: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
):
    chat_message_history = SQLChatMessageHistory(
        session_id=chat_history,
        connection=engine
    )
    return chat_message_history.messages


@router.post('/chat')
async def chat(
        session: SessionDep,
        request: ChatRequest,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    logger.debug(f'knowledge_base: {request.knowledge_base_name} session:{request.history}')
    logger.info(f'{current_user.username}: {request.message}')

    async def generate():
        # 发送模型加载状态
        yield SSEMessage(
            event=ChatEventType.STATUS,
            data="正在加载模型..."
        ).to_sse()
        
        await asyncio.sleep(0.1)  # 添加小延迟
        
        embedding = load_embedding()
        reranker = load_reranker()
        vec_db = get_vector_db(request.knowledge_base_name, embedding, db_name='llm_chat')
        doc_db = get_doc_db(request.knowledge_base_name)

        # 发送文档检索状态
        yield SSEMessage(
            event=ChatEventType.STATUS,
            data="正在检索相关文档..."
        ).to_sse()
        
        await asyncio.sleep(0.1)  # 添加小延迟

        retriever = base_retriever(vec_db, doc_db, reranker)
        docs = retriever.invoke(request.message)

        # 发送检索到的文档
        docs_data = [{
            'content': doc.page_content,
            'metadata': doc.metadata
        } for doc in docs]
        yield SSEMessage(
            event=ChatEventType.DOCS,
            data=docs_data
        ).to_sse()
        
        await asyncio.sleep(0.1)  # 添加小延迟

        # 发送生成回答状态
        yield SSEMessage(
            event=ChatEventType.STATUS,
            data="正在生成回答..."
        ).to_sse()

        chain = rag_chat(request.message, docs)
        full_response = ''
        
        async for chunk in chain.astream({
            'chat_history': [],
            'docs': docs,
            'question': request.message
        }):
            if chunk.content:
                full_response += chunk.content
                yield SSEMessage(
                    event=ChatEventType.ANSWER,
                    data=chunk.content
                ).to_sse()
                
        logger.info(f'AI: {full_response}')

    # 更新用户信息
    update_user(
        session,
        str(current_user.email),
        UserUpdate(last_chat=request.history, last_project=request.knowledge_base_name)
    )

    return StreamingResponse(
        generate(),
        media_type='text/event-stream'
    )
