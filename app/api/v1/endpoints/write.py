import asyncio
import json
from datetime import datetime
from enum import Enum
from operator import itemgetter
from typing import Annotated, Union
from uuid import uuid4
import os

from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, trim_messages
from loguru import logger
from pydantic import BaseModel
from starlette.responses import StreamingResponse

import app.crud.chat_record as record_crud
import app.crud.chat_session as chat_crud
import app.crud.manuscript as manu_crud
import app.crud.project_source as source_crud
import app.crud.user as user_crud
import app.crud.write_project as project_crud
from app.core.config import get_settings
from app.core.security import get_current_active_user
from app.db.session import SessionDep, engine
from app.models import UserTable, WriteProjectTable, ChatSessionTable, ManuscriptTable
from app.models.write_project import ChatRecordTable, ProjectSourcesTable
from app.schemas.chat_session import ChatSession, ChatSessionUpdate
from app.schemas.manuscript import ManuscriptPublic, Manuscript, ManuscriptType
from app.schemas.user import UserUpdate
from app.schemas.write_project import WriteProject, WriteProjectUpdate
from llm.core.agent import MainAgent, MainAgentState
from llm.core.chain import conclude_chat
from llm.core.model import load_llm
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
        return f'event: {self.event.value}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n'


class PDFInfo(BaseModel):
    file_name: str
    file_size: int
    file_url: str
    upload_time: datetime


@router.post('/projects', response_model=WriteProject, description='新建写作工程')
async def add_new_project(
    session: SessionDep,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
    description: str = Form(...),
):
    now_time = datetime.now()

    new_project = WriteProjectTable(
        uid=str(uuid4()),
        description=description,
        user_email=str(current_user.email),
        create_time=now_time,
        update_time=now_time,
    )

    manuscript = ManuscriptTable(
        uid=str(uuid4()),
        project_uid=new_project.uid,
        title='01-未命名文件',
        file_type=ManuscriptType.CONTEXT,
    )

    manuscript = manu_crud.insert(session, manuscript)
    new_project.last_manuscript = manuscript.uid

    new_project = project_crud.insert(session, new_project)
    return new_project


@router.delete('/projects/{project_uid}', description='删除指定项目')
async def delete_project(
    session: SessionDep,
    project_uid: str,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    project_crud.delete(session, project_uid)

    if current_user.last_project == project_uid:
        user_crud.update(session, str(current_user.email), UserUpdate(last_project=''))

    # TODO 视情况是否要删除其他东西
    chat_sessions = chat_crud.get_list(session, project_uid, str(current_user.email))
    for chat_session in chat_sessions:
        chat_message_history = SQLChatMessageHistory(
            session_id=chat_session.uid, connection=engine
        )
        chat_message_history.clear()
        chat_crud.delete(session, chat_session.uid)

    manu_crud.delete_by_parent(session, project_uid)
    source_crud.delete_by_parent(session, project_uid)


@router.get(
    '/projects', description='返回知识库列表', response_model=list[WriteProject]
)
async def read_projects(
    session: SessionDep,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
):
    return project_crud.get_list(session, str(current_user.email))


@router.post('/projects/{project_uid}/manuscripts', description='新建草稿')
async def add_new_manuscript(
    session: SessionDep,
    project_uid: str,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
    title: str = Form(...),
) -> str:
    manuscript = ManuscriptTable(
        uid=str(uuid4()),
        project_uid=project_uid,
        title=title,
    )

    manuscript = manu_crud.insert(session, manuscript)
    return manuscript.uid


@router.patch('/projects/{project_uid}/manuscripts/{uid}')
async def save_manuscript(
    session: SessionDep,
    project_uid: str,
    uid: str,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
    content: str = Form(...),
) -> str:
    now_time = datetime.now()
    last_manuscript = manu_crud.get_manuscript(session, uid)

    if last_manuscript.content == content:
        return uid

    new_manuscript = ManuscriptTable(
        uid=uid,
        project_uid=last_manuscript.project_uid,
        title=last_manuscript.title,
        content=content,
        version=last_manuscript.version + 1,
        file_type=last_manuscript.file_type,
    )
    new_manuscript = manu_crud.insert(session, new_manuscript)

    project = WriteProjectUpdate(last_manuscript=uid, update_time=now_time)
    project_crud.update(session, new_manuscript.project_uid, project)

    return new_manuscript.uid


@router.get(
    '/projects/{project_uid}/manuscripts',
    response_model=list[ManuscriptPublic],
    description='获取指定项目的草稿列表',
)
async def get_manuscripts(
    session: SessionDep,
    project_uid: str,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    return manu_crud.get_list(session, project_uid)


@router.get(
    '/projects/{project_uid}/manuscripts/{uid}',
    response_model=Manuscript,
    description='查询指定草稿内容',
)
async def read_manuscript(
    session: SessionDep,
    project_uid: str,
    uid: str,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    return manu_crud.get_manuscript(session, uid)


@router.get(
    '/projects/{project_uid}/sources',
    response_model=list[Document],
    description='获取指定项目的索引列表',
)
async def get_project_sources(
    session: SessionDep,
    project_uid: str,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    project_sources = source_crud.get_list(session, project_uid)
    return project_sources.get_sources() if project_sources else []


@router.post('/projects/{project_uid}/chats', description='新建对话')
async def insert_chat(
    session: SessionDep,
    project_uid: str,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
) -> str:
    now_time = datetime.now()

    new_chat = ChatSessionTable(
        uid=str(uuid4()),
        parent_uid=project_uid,
        description='新建对话',
        user_email=str(current_user.email),
        create_time=now_time,
        update_time=now_time,
    )

    new_chat = chat_crud.insert(session, new_chat)
    return new_chat.uid


@router.delete('/projects/{project_uid}/chats/{chat_uid}', description='删除指定对话')
async def delete_chat(
    session: SessionDep,
    project_uid: str,
    chat_uid: str,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    chat_crud.delete(session, chat_uid)
    chat_message_history = SQLChatMessageHistory(session_id=chat_uid, connection=engine)
    chat_message_history.clear()


@router.get(
    '/projects/{project_uid}/chats',
    response_model=list[ChatSession],
    description='获取指定项目的所有对话',
)
async def get_chats(
    session: SessionDep,
    project_uid: str,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    return chat_crud.get_list(session, project_uid, str(current_user.email))


@router.get(
    '/projects/{project_uid}/chats/{chat_uid}/messages', description='加载历史对话'
)
async def get_chat(
    project_uid: str,
    chat_uid: str,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    chat_message_history = SQLChatMessageHistory(session_id=chat_uid, connection=engine)
    return chat_message_history.messages


@router.post(
    '/projects/{project_uid}/chats/{chat_uid}/messages', description='请求对话'
)
async def chat(
    session: SessionDep,
    project_uid: str,
    chat_uid: str,
    current_user: Annotated[UserTable, Depends(get_current_active_user)],
    message: str = Form(...),
    current_text: str = Form(None),
    files: list[UploadFile] = File([]),
    model: str = Form(...),
    temperature: float = Form(...),
    context_length: int = Form(...),
    use_web: bool = Form(...),
    available_kbs: str = Form('[]'),
):
    available_kbs = json.loads(available_kbs)

    logger.debug(available_kbs)
    logger.debug(f'project: {project_uid} session:{chat_uid}')
    logger.info(f'{current_user.username}: {message}')
    user_email = current_user.email

    chat_message_history = SQLChatMessageHistory(session_id=chat_uid, connection=engine)

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

                uploaded_files.append(
                    {
                        'original_name': file.filename,
                        'saved_name': unique_filename,
                        'path': str(file_path),
                    }
                )

            except Exception as e:
                logger.error(f'Error processing file {file.filename}: {str(e)}')
                raise HTTPException(
                    status_code=500, detail=f'处理文件时发生错误 {file.filename}'
                ) from e

    async def event_generator():
        try:
            if uploaded_files:
                yield SSEMessage(event=ChatEventType.STATUS, data='文件已收到...').to_sse()

            if not message:
                yield SSEMessage(
                    event=ChatEventType.ANSWER, data='文件已收到，请给出你的需求'
                ).to_sse()
            else:
                yield SSEMessage(event=ChatEventType.STATUS, data='唤醒智能体').to_sse()
                await asyncio.sleep(0.1)

                cut_messages = trim_messages(
                    chat_message_history.messages,
                    strategy='last',
                    token_counter=len,
                    max_tokens=context_length,
                    start_on='human',
                    end_on=('ai', 'tool'),
                    include_system=False,
                )
                cut_messages.append(HumanMessage(content=message))

                llm = load_llm(model, temperature)
                app = MainAgent(
                    use_web=use_web, task_llm=llm, available_knowledge_bases=available_kbs
                ).build()
                old_source_data = source_crud.get_list(session, project_uid)
                if old_source_data:
                    old_source = [
                        (doc.metadata['file_id'], doc)
                        for doc in old_source_data.get_sources()
                    ]
                else:
                    old_source = []
                full_response = ''
                await asyncio.sleep(0.1)

                async for event in app.astream_events(
                    {
                        'messages': cut_messages,
                        'current_text': current_text,
                        'project_uid': project_uid,
                        'sources': old_source,
                    },
                    {'configurable': {'thread_id': '1'}, 'recursion_limit': 25},
                    version='v2',
                    exclude_names=['_write', 'RunnableSequence', 'RunnableLambda'],
                ):
                    if event['event'] == 'on_chat_model_stream':
                        if (
                            event['metadata']['langgraph_node']
                            and event['metadata']['langgraph_node'] == 'main_router'
                        ):
                            data = event['data']
                            if data['chunk'].content:
                                full_response += data['chunk'].content
                                yield SSEMessage(
                                    event=ChatEventType.ANSWER,
                                    data=data['chunk'].content,
                                ).to_sse()
                        elif event['metadata']['langgraph_node'] and event['metadata'][
                            'langgraph_node'
                        ] in ['generator', 'rewriter']:
                            data = event['data']
                            if data['chunk'].content:
                                yield SSEMessage(
                                    event=ChatEventType.WRITE,
                                    data=data['chunk'].content,
                                ).to_sse()

                    elif event['event'] == 'on_chain_end':
                        if event['name'] == 'main_route':
                            yield SSEMessage(
                                event=ChatEventType.STATUS, data='chat_end'
                            ).to_sse()
                        elif (
                            event['name'] == 'LangGraph'
                            and len(event['metadata'].keys()) == 1
                        ):
                            now_time = datetime.now()
                            final_output: MainAgentState = event['data']['output']

                            # 保存本次调用记录
                            chat_record = ChatRecordTable(
                                project_uid=project_uid,
                                user_email=str(user_email),
                                price=final_output['price'],
                                create_time=now_time,
                            )
                            record_crud.insert(session, chat_record)

                            # 更新引用源
                            new_sources = ProjectSourcesTable(
                                project_uid=project_uid,
                            )
                            new_sources.set_sources(
                                list(map(itemgetter(1), final_output['sources']))
                            )
                            source_crud.insert_or_update(session, new_sources)

                    elif event['event'] == 'on_tool_end':
                        if event['name'] == 'modifier':
                            modify: OptimizerOutput = event['data']['output']
                            yield SSEMessage(
                                event=ChatEventType.MODIFY, data=modify.model_dump()
                            ).to_sse()

                    elif event['event'] == 'on_tool_start':
                        if event['name'] == 'select_vecstore':
                            yield SSEMessage(
                                event=ChatEventType.STATUS, data='自行决策选择知识库'
                            ).to_sse()
                        elif event['name'] == 'search_from_vecstore':
                            if event['metadata']['langgraph_node'] == 'rag_paper_search':
                                yield SSEMessage(
                                    event=ChatEventType.STATUS, data='搜索相关文献'
                                ).to_sse()
                            else:
                                yield SSEMessage(
                                    event=ChatEventType.STATUS, data='搜索知识库'
                                ).to_sse()
                        elif event['name'] == 'modifier':
                            yield SSEMessage(
                                event=ChatEventType.STATUS, data='分析修改策略'
                            ).to_sse()
                        elif event['name'] == 'rewriter':
                            yield SSEMessage(
                                event=ChatEventType.WRITE,
                                data='\n\n---\n\n',
                            ).to_sse()
                            yield SSEMessage(
                                event=ChatEventType.STATUS, data='修改文本'
                            ).to_sse()
                        else:
                            yield SSEMessage(
                                event=ChatEventType.STATUS, data=event['name']
                            ).to_sse()

                    elif event['event'] == 'on_chain_start':
                        if event['name'] == 'search_conclude':
                            yield SSEMessage(
                                event=ChatEventType.STATUS, data='总结搜索结果'
                            ).to_sse()
                        elif event['name'] == 'analyzer':
                            yield SSEMessage(
                                event=ChatEventType.STATUS, data='整理写作资料'
                            ).to_sse()
                        elif event['name'] == 'generator':
                            yield SSEMessage(
                                event=ChatEventType.STATUS, data='文本创作'
                            ).to_sse()

                logger.info(f'{model}: {full_response}')
                chat_message_history.add_user_message(message)
                chat_message_history.add_ai_message(full_response)

        except Exception as e:
            logger.exception('chat error')
            yield SSEMessage(event=ChatEventType.ERROR, data=str(e)).to_sse()

        finally:
            # 清理临时文件
            for file_info in uploaded_files:
                try:
                    clean_path = file_info['path']
                    if os.path.exists(clean_path):
                        os.remove(clean_path)
                        logger.debug(f'Cleaned up temporary file: {clean_path}')
                except Exception as e:
                    logger.error(f'Error cleaning up file: {str(e)}')

    async def generate_summary():
        conclude = conclude_chat(chat_message_history)
        now_time = datetime.now()
        chat_crud.update(
            session,
            chat_uid,
            ChatSessionUpdate(description=conclude.content, update_time=now_time),
        )

    # 自动生成总结
    chat_info = chat_crud.get(session, chat_uid)
    if (
        chat_info
        and chat_info.description == '新建对话'
        and chat_message_history.messages
    ):
        asyncio.create_task(generate_summary())

    # 更新用户信息
    user_crud.update(
        session, str(current_user.email), UserUpdate(last_project=project_uid)
    )
    return StreamingResponse(event_generator(), media_type='text/event-stream')
