import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import Annotated, List, Union, Any
from uuid import uuid4
import os

from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.core.config import get_settings
from app.core.security import get_current_active_user
from app.crud.chat_session import insert_chat, get_chat_list, update_chat, get_chat
from app.crud.manuscript import insert_manuscript, get_manuscripts_list, get_manuscript
from app.crud.user import update_user
from app.crud.write_project import insert_project, get_project_list, get_project
from app.db.session import SessionDep, engine
from app.models import UserTable, WriteProjectTable, ChatSessionTable, ManuscriptTable
from app.schemas.chat_session import ChatSession, ChatSessionUpdate
from app.schemas.manuscript import ManuscriptPublic, Manuscript
from app.schemas.user import User, UserUpdate
from app.schemas.write_project import WriteProject
from llm.core.agent import MainAgent
from llm.core.chain import conclude_chat
from llm.core.model import load_glm4_flash, load_gpt4o_mini
from llm.tool.modify import OptimizerOutput

router = APIRouter()


class ChatEventType(Enum):
    STATUS = 'status'
    DOCS = 'docs'
    MODIFY = 'modify'
    ANSWER = 'answer'
    WRITE = 'write'
    ERROR = 'error'


class SSEMessage(BaseModel):
    event: ChatEventType
    data: Union[str, list[dict], dict]

    def to_sse(self) -> str:
        """转换为 SSE 格式的消息"""
        return f"event: {self.event.value}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


@router.get(
    '/projects',
    description='返回知识库列表',
    response_model=list[WriteProject]
)
async def read_projects(
        session: SessionDep,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
        offset: int = 0,
        limit: Annotated[int, Query(le=20)] = 20,

):
    return get_project_list(session, current_user.email)


@router.patch('/new_project', description='新建写作工程')
async def add_new_project(
        session: SessionDep,
        description: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
) -> str:
    now_time = datetime.now()

    new_project = WriteProjectTable(
        uid=str(uuid4()),
        graph_checkpoint=str(uuid4()),
        description=description,
        user_email=str(current_user.email),
        create_time=now_time,
        update_time=now_time
    )

    new_project = insert_project(session, new_project)
    return new_project.uid


@router.get('/chats', response_model=list[ChatSession])
async def get_chats(
        session: SessionDep,
        project_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
):
    return get_chat_list(session, project_uid)


@router.patch('/new_chat')
async def add_new_chat(
        session: SessionDep,
        project_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
) -> str:
    now_time = datetime.now()

    new_chat = ChatSessionTable(
        chat_uid=str(uuid4()),
        parent_uid=project_uid,
        description='新建对话',
        user_email=str(current_user.email),
        # user_email='admin@example.com',
        create_time=now_time,
        update_time=now_time
    )

    new_chat = insert_chat(session, new_chat)
    return new_chat.chat_uid


@router.get('/manuscripts', response_model=list[ManuscriptPublic])
async def get_manuscripts(
        session: SessionDep,
        project_uid: str,
        # current_user: Annotated[UserTable, Depends(get_current_active_user)]
):
    return get_manuscripts_list(session, project_uid)


@router.get('/manuscript', response_model=Manuscript)
async def read_manuscript(
        session: SessionDep,
        uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
):
    return get_manuscript(session, uid)


@router.patch('/new_manuscript')
async def add_new_manuscript(
        session: SessionDep,
        project_uid: str,
        title: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
) -> str:
    manuscript = ManuscriptTable(
        uid=str(uuid4()),
        project_uid=project_uid,
        title=title,
    )

    manuscript = insert_manuscript(session, manuscript)
    return manuscript.uid


@router.post('/chat')
async def chat(
        session: SessionDep,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
        project_uid: str = Form(...),
        chat_uid: str = Form(...),
        message: str = Form(None),
        current_text: str = Form(None),
        files: list[UploadFile] = File([])
):
    logger.debug(f'project: {project_uid} session:{chat_uid}')

    chat_message_history = SQLChatMessageHistory(
        session_id=chat_uid,
        connection=engine
    )

    uploaded_files = []
    if files:
        file_output_parent = get_settings().server.TEMP_DIR
        for file in files:
            try:
                # 生成唯一的文件名
                file_ext = os.path.splitext(file.filename)[1]
                unique_filename = f'{uuid4()}{file_ext}'
                file_path = os.path.join(file_output_parent, unique_filename)

                # 写入到新文件
                with open(file_path, 'wb') as f:
                    contents = await file.read()
                    f.write(contents)

                uploaded_files.append({
                    'original_name': file.filename,
                    'saved_name': unique_filename,
                    'path': str(file_path)
                })

            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"处理文件时发生错误 {file.filename}") from e

    async def generate_summary():
        conclude = conclude_chat(chat_message_history)
        now_time = datetime.now()
        update_chat(
            session,
            chat_uid,
            ChatSessionUpdate(
                description=conclude.content,
                update_time=now_time
            )
        )

    # 自动生成总结
    chat_info = get_chat(session, chat_uid)
    if chat_info.description == '新建对话' and chat_message_history.messages:
        asyncio.create_task(generate_summary())

    # 更新用户信息
    update_user(
        session,
        str(current_user.email),
        UserUpdate(last_project=project_uid)
    )

    return StreamingResponse(
        event_generator(message, uploaded_files, current_text, current_user),
        media_type='text/event-stream'
    )


async def event_generator(
        message: str,
        uploaded_files: list[dict[str, Any]],
        current_text: str,
        user: User
):
    try:
        if uploaded_files:
            yield SSEMessage(
                event=ChatEventType.STATUS,
                data='文件已收到...'
            ).to_sse()

        if not message:
            yield SSEMessage(
                event=ChatEventType.ANSWER,
                data='文件已收到，请给出你的需求'
            ).to_sse()
        else:
            logger.info(f'{user.username}: {message}')
            llm = load_gpt4o_mini()
            app = MainAgent(llm=llm).build()

            async for event in app.astream_events(
                    {
                        'messages': [HumanMessage(content=message)],
                        'current_text': current_text
                    },
                    {'configurable': {'thread_id': '1'}, 'recursion_limit': 25},
                    version='v2',
                    exclude_names=['_write', 'RunnableSequence', 'RunnableLambda']
            ):
                if event['event'] == 'on_chat_model_stream' and event['metadata']['langgraph_node'] == 'main_router':
                    data = event['data']
                    if data['chunk'].content:
                        yield SSEMessage(
                            event=ChatEventType.ANSWER,
                            data=data['chunk'].content
                        ).to_sse()

                if event['event'] == 'on_chat_model_end':
                    yield SSEMessage(
                        event=ChatEventType.STATUS,
                        data='chat end'
                    ).to_sse()

                if event['event'] == 'on_tool_end' and event['name'] == 'modifier':
                    modify: OptimizerOutput = event['data']['output']['parsed']
                    yield SSEMessage(
                        event=ChatEventType.MODIFY,
                        data=modify.model_dump()
                    ).to_sse()

                # if event['event'] == 'on_chat_model_stream' and event['metadata']['langgraph_node'] == 'search_conclude':
                #     data = event['data']
                #     if data['chunk'].content:
                #         yield SSEMessage(
                #             event=ChatEventType.WRITE,
                #             data=data['chunk'].content
                #         ).to_sse()

    except Exception as e:
        logger.error(f"Unexpected error in event generator: {str(e)}")
        yield SSEMessage(
            event=ChatEventType.ERROR,
            data=str(e)
        ).to_sse()

    finally:
        # 清理临时文件
        for file_info in uploaded_files:
            try:
                file_path = file_info['path']
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {str(e)}")
