from datetime import datetime
from typing import Annotated, Union
from uuid import uuid4
from enum import Enum
import json
import asyncio

from fastapi import APIRouter, Query, Depends, Form
from langchain_community.chat_message_histories import SQLChatMessageHistory
from loguru import logger
from pydantic import BaseModel
from starlette.responses import StreamingResponse

import app.crud.chat_session as chat_crud
from app.core.security import get_current_active_user
from app.crud.knowledge_base import get_knowledge_bases, get_knowledge_base
from app.crud.user import update_user
from app.db.session import SessionDep, engine
from app.models import UserTable, ChatSessionTable
from app.schemas.chat_session import ChatSession, ChatSessionUpdate
from app.schemas.knowledge_base import KnowledgeBase
from app.schemas.user import UserUpdate
from llm.core.chain import rag_chain, conclude_chat
from llm.core.model import load_embedding, load_reranker
from llm.rag.retriever import base_retriever
from llm.rag.storage import get_vector_db, get_doc_db


class ChatEventType(Enum):
    STATUS = 'status'
    DOCS = 'docs'
    ANSWER = 'answer'


class SSEMessage(BaseModel):
    event: ChatEventType
    data: Union[str, list[dict], dict]

    def to_sse(self) -> str:
        """转换为 SSE 格式的消息"""
        return f"event: {self.event.value}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


router = APIRouter()


@router.get(
    '/knowledge_bases',
    response_model=list[KnowledgeBase],
    description='返回知识库列表',
)
async def read_knowledge_bases(
        session: SessionDep,
        offset: int = 0,
        limit: Annotated[int, Query(le=20)] = 20,
):
    return get_knowledge_bases(session, offset, limit)


@router.post(
    '/knowledge_bases/{knowledge_base_uid}/chats',
    description='在对应知识库下新建对话'
)
async def insert_chat(
        session: SessionDep,
        knowledge_base_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
) -> str:
    now_time = datetime.now()

    new_chat = ChatSessionTable(
        uid=str(uuid4()),
        parent_uid=knowledge_base_uid,
        user_email=str(current_user.email),
        create_time=now_time,
        update_time=now_time
    )
    new_chat = chat_crud.insert(session, new_chat)
    return new_chat.uid


@router.get(
    '/knowledge_bases/{knowledge_base_uid}/chats',
    response_model=list[ChatSession],
    description='获取对应知识库下的对话列表'
)
async def get_chats(
        session: SessionDep,
        knowledge_base_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
):
    return chat_crud.get_list(session, knowledge_base_uid)


@router.delete(
    '/knowledge_bases/{knowledge_base_uid}/chats/{chat_uid}',
    description='删除指定对话历史记录'
)
async def delete_chat(
        session: SessionDep,
        chat_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    chat_crud.delete(session, chat_uid)
    chat_message_history = SQLChatMessageHistory(
        session_id=chat_uid,
        connection=engine
    )
    chat_message_history.clear()


@router.get(
    '/knowledge_bases/{knowledge_base_uid}/chats/{chat_uid}',
    response_model=ChatSession,
    description='返回对话信息'
)
async def get_chat(
        session: SessionDep,
        chat_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
):
    return chat_crud.get(session, chat_uid)


@router.get(
    '/knowledge_bases/{knowledge_base_uid}/chats/{chat_uid}/messages',
    description='加载对话历史'
)
async def get_chat_history(
        chat_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
):
    chat_message_history = SQLChatMessageHistory(
        session_id=chat_uid,
        connection=engine
    )
    return chat_message_history.messages


@router.post(
    '/knowledge_bases/{knowledge_base_uid}/chats/{chat_uid}/messages',
    description='请求对话'
)
async def chat(
        session: SessionDep,
        knowledge_base_uid: str,
        chat_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
        message: str = Form(...),
):
    chat_message_history = SQLChatMessageHistory(
        session_id=chat_uid,
        connection=engine
    )
    knowledge_base = get_knowledge_base(session, knowledge_base_uid)
    table_name = knowledge_base.table_name

    logger.debug(f'knowledge_base: {table_name} session:{chat_uid}')
    logger.info(f'{current_user.username}: {message}')

    async def generate():
        # 发送模型加载状态
        yield SSEMessage(
            event=ChatEventType.STATUS,
            data='正在加载模型...'
        ).to_sse()

        await asyncio.sleep(0.1)  # 添加小延迟

        embedding = load_embedding()
        reranker = load_reranker()
        vec_db = get_vector_db(table_name, embedding, db_name='llm_chat')
        doc_db = get_doc_db(table_name)

        # 发送文档检索状态
        yield SSEMessage(
            event=ChatEventType.STATUS,
            data='正在检索相关文档...'
        ).to_sse()

        await asyncio.sleep(0.1)  # 添加小延迟

        retriever = base_retriever(vec_db, doc_db, reranker)
        docs = retriever.invoke(message)

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
            data='正在生成回答...'
        ).to_sse()

        chain = rag_chain()
        full_response = ''

        async for chunk in chain.astream({
            'chat_history': [],
            'docs': docs,
            'question': message
        }):
            if chunk.content:
                full_response += chunk.content
                yield SSEMessage(
                    event=ChatEventType.ANSWER,
                    data=chunk.content
                ).to_sse()

        logger.info(f'AI: {full_response}')
        chat_message_history.add_user_message(message)
        chat_message_history.add_ai_message(full_response)

    async def generate_summary():
        conclude = conclude_chat(chat_message_history)
        now_time = datetime.now()
        chat_crud.update(
            session,
            chat_uid,
            ChatSessionUpdate(
                description=conclude.content,
                update_time=now_time
            )
        )

    # 自动生成总结
    chat_info = chat_crud.get(session, chat_uid)
    if chat_info.description == '新建对话' and chat_message_history.messages:
        asyncio.create_task(generate_summary())

    # 更新用户信息
    update_user(
        session,
        str(current_user.email),
        UserUpdate(last_knowledge_base=knowledge_base_uid)
    )

    return StreamingResponse(
        generate(),
        media_type='text/event-stream'
    )
