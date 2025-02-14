import asyncio
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Union
from uuid import uuid4
import os

from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, trim_messages
from loguru import logger
from pydantic import BaseModel
from starlette.responses import StreamingResponse, FileResponse

import app.crud.chat_session as chat_crud
from app.core.config import get_settings
from app.core.security import get_current_active_user
from app.crud.manuscript import insert_manuscript, get_manuscripts_list, get_manuscript
from app.crud.user import update_user
from app.crud.write_project import insert_project, get_project_list, get_project, update_project
from app.db.session import SessionDep, engine
from app.models import UserTable, WriteProjectTable, ChatSessionTable, ManuscriptTable
from app.schemas.chat_session import ChatSession, ChatSessionUpdate
from app.schemas.manuscript import ManuscriptPublic, Manuscript, ManuscriptUpdate
from app.schemas.user import UserUpdate
from app.schemas.write_project import WriteProject, WriteProjectUpdate
from llm.core.agent import MainAgent
from llm.core.chain import conclude_chat
from llm.tool.modify import OptimizerOutput, RewriterOutput

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


class PDFInfo(BaseModel):
    file_name: str
    file_size: int
    file_url: str
    upload_time: datetime


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


@router.patch('/new_chat')
async def insert_chat(
        session: SessionDep,
        project_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
) -> str:
    now_time = datetime.now()

    new_chat = ChatSessionTable(
        uid=str(uuid4()),
        parent_uid=project_uid,
        description='新建对话',
        user_email=str(current_user.email),
        # user_email='admin@example.com',
        create_time=now_time,
        update_time=now_time
    )

    new_chat = chat_crud.insert(session, new_chat)
    return new_chat.uid


@router.delete('/delete_chat/{chat_uid}')
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


@router.get('/chats', response_model=list[ChatSession])
async def get_chats(
        session: SessionDep,
        project_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
):
    return chat_crud.get_list(session, project_uid)


@router.get('/chat/{chat_uid}')
async def get_chat(
        chat_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
):
    chat_message_history = SQLChatMessageHistory(
        session_id=chat_uid,
        connection=engine
    )
    return chat_message_history.messages


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


@router.post('/save_manuscript')
async def save_manuscript(
        session: SessionDep,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
        uid: str = Form(...),
        content: str = Form(...)
) -> str:
    now_time = datetime.now()
    last_manuscript = get_manuscript(session, uid)

    if last_manuscript.content == content:
        return uid

    new_manuscript = ManuscriptTable(
        uid=uid,
        project_uid=last_manuscript.project_uid,
        title=last_manuscript.title,
        content=content,
        version=last_manuscript.version + 1,
        is_draft=last_manuscript.is_draft
    )
    new_manuscript = insert_manuscript(session, new_manuscript)

    project = WriteProjectUpdate(
        last_manuscript=uid,
        update_time=now_time
    )
    update_project(session, new_manuscript.project_uid, project)

    return new_manuscript.uid


@router.post('/chat')
async def chat(
        session: SessionDep,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
        project_uid: str = Form(...),
        graph_ckpt: str = Form(...),
        chat_uid: str = Form(...),
        message: str = Form(...),
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

    async def event_generator():
        # try:
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
            logger.info(f'{current_user.username}: {message}')
            yield SSEMessage(
                event=ChatEventType.STATUS,
                data='唤醒智能体'
            ).to_sse()

            await asyncio.sleep(0.1)

            cut_messages = trim_messages(
                chat_message_history.messages,
                strategy='last',
                token_counter=len,
                max_tokens=5,
                start_on='human',
                end_on='ai',
                include_system=False
            )
            cut_messages.append(HumanMessage(content=message))

            app = MainAgent(use_web=False).build()
            full_response = ''
            await asyncio.sleep(0.1)

            async for event in app.astream_events(
                    {
                        'messages': cut_messages,
                        'current_text': current_text,
                        'project_uid': project_uid
                    },
                    {'configurable': {'thread_id': '1'}, 'recursion_limit': 25},
                    version='v2',
                    exclude_names=['_write', 'RunnableSequence', 'RunnableLambda']
            ):
                if event['event'] == 'on_chat_model_stream':
                    if event['metadata']['langgraph_node'] and event['metadata']['langgraph_node'] == 'main_router':
                        data = event['data']
                        if data['chunk'].content:
                            full_response += data['chunk'].content
                            yield SSEMessage(
                                event=ChatEventType.ANSWER,
                                data=data['chunk'].content
                            ).to_sse()
                    elif event['metadata']['langgraph_node'] and event['metadata']['langgraph_node'] == 'generator':
                        data = event['data']
                        if data['chunk'].content:
                            yield SSEMessage(
                                event=ChatEventType.WRITE,
                                data=data['chunk'].content
                            ).to_sse()

                elif event['event'] == 'on_chain_end':
                    if event['name'] == 'main_route':
                        yield SSEMessage(
                            event=ChatEventType.STATUS,
                            data='chat_end'
                        ).to_sse()

                elif event['event'] == 'on_tool_end':
                    if event['name'] == 'modifier':
                        modify: OptimizerOutput = event['data']['output']['parsed']
                        yield SSEMessage(
                            event=ChatEventType.MODIFY,
                            data=modify.model_dump()
                        ).to_sse()
                    elif event['name'] == 'rewriter':
                        output: RewriterOutput = event['data']['output']['parsed']
                        yield SSEMessage(
                            event=ChatEventType.WRITE,
                            data=f'\n\n---\n\n{output.rewrite}'
                        ).to_sse()

                elif event['event'] == 'on_tool_start':
                    if event['name'] == 'select_vecstore':
                        yield SSEMessage(
                            event=ChatEventType.STATUS,
                            data='自行决策选择知识库'
                        ).to_sse()
                    elif event['name'] == 'search_from_vecstore':
                        yield SSEMessage(
                            event=ChatEventType.STATUS,
                            data='搜索知识库'
                        ).to_sse()
                    elif event['name'] == 'modifier':
                        yield SSEMessage(
                            event=ChatEventType.STATUS,
                            data='分析修改策略'
                        ).to_sse()
                    else:
                        yield SSEMessage(
                            event=ChatEventType.STATUS,
                            data=event['name']
                        ).to_sse()

                elif event['event'] == 'on_chain_start':
                    if event['name'] == 'search_conclude':
                        yield SSEMessage(
                            event=ChatEventType.STATUS,
                            data='总结搜索结果'
                        ).to_sse()
                    elif event['name'] == 'analyzer':
                        yield SSEMessage(
                            event=ChatEventType.STATUS,
                            data='整理写作资料'
                        ).to_sse()
                    elif event['name'] == 'generator':
                        yield SSEMessage(
                            event=ChatEventType.STATUS,
                            data='文本创作'
                        ).to_sse()

            logger.info(f'AI: {full_response}')
            chat_message_history.add_user_message(message)
            chat_message_history.add_ai_message(full_response)

        # except Exception as e:
        #     logger.error(f"Unexpected error in event generator: {str(e)}")
        #     yield SSEMessage(
        #         event=ChatEventType.ERROR,
        #         data=str(e)
        #     ).to_sse()
        #
        # finally:
        #     # 清理临时文件
        #     for file_info in uploaded_files:
        #         try:
        #             file_path = file_info['path']
        #             if os.path.exists(file_path):
        #                 os.remove(file_path)
        #                 logger.debug(f"Cleaned up temporary file: {file_path}")
        #         except Exception as e:
        #             logger.error(f"Error cleaning up file {file_path}: {str(e)}")

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
    if chat_info and chat_info.description == '新建对话' and chat_message_history.messages:
        asyncio.create_task(generate_summary())

    # 更新用户信息
    update_user(
        session,
        str(current_user.email),
        UserUpdate(last_project=project_uid)
    )
    return StreamingResponse(
        event_generator(),
        media_type='text/event-stream'
    )


@router.get(
    '/pdf_files',
    description='获取可供阅读的PDF文件列表',
    response_model=list[PDFInfo]
)
async def get_pdf_files(
        session: SessionDep,
        project_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
        offset: int = 0,
        limit: Annotated[int, Query(le=20)] = 20,
):
    """
    获取项目中的PDF文件列表
    """
    settings = get_settings()
    pdf_dir = os.path.join(settings.server.TEMP_DIR, project_uid)

    if not os.path.exists(pdf_dir):
        return []

    pdf_files = []
    for file_name in os.listdir(pdf_dir):
        if file_name.lower().endswith('.pdf'):
            file_path = os.path.join(pdf_dir, file_name)
            file_stat = os.stat(file_path)

            # 构建文件URL
            file_url = f"/api/v1/write/pdf/{project_uid}/{file_name}"

            pdf_files.append(PDFInfo(
                file_name=file_name,
                file_size=file_stat.st_size,
                file_url=file_url,
                upload_time=datetime.fromtimestamp(file_stat.st_mtime)
            ))

    # 按上传时间倒序排序
    pdf_files.sort(key=lambda x: x.upload_time, reverse=True)

    # 分页
    start = offset
    end = offset + limit
    return pdf_files[start:end]


@router.get(
    '/pdf',
    description='获取PDF文件内容'
)
async def get_pdf_file(
        file_path: str = Form(...)
):
    """
    获取PDF文件内容
    """

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        file_path,
        media_type='application/pdf',
        filename=Path(file_path).name
    )
