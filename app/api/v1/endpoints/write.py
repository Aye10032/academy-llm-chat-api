import json
from datetime import datetime
from enum import Enum
from typing import Annotated, List, Union
from uuid import uuid4
import os

from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException
from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.core.security import get_current_active_user
from app.crud.chat_session import insert_chat, get_chat_list
from app.crud.manuscript import insert_manuscript, get_manuscripts_list, get_manuscript
from app.crud.write_project import insert_project, get_project_list, get_project
from app.db.session import SessionDep
from app.models import UserTable, WriteProjectTable, ChatSessionTable, ManuscriptTable
from app.schemas.chat_session import ChatSession
from app.schemas.manuscript import ManuscriptPublic, Manuscript
from app.schemas.user import User
from app.schemas.write_project import WriteProject
from llm.core.agent import MainAgent
from llm.core.model import load_glm4_flash, load_gpt4o_mini
from llm.tool.modify import OptimizerOutput

router = APIRouter()


class ChatEventType(Enum):
    STATUS = 'status'
    DOCS = 'docs'
    MODIFY = 'modify'
    ANSWER = 'answer'


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


async def event_generator(
        message: str,
        files: list[UploadFile],
        current_text: str,
        user: User
):
    file_output_parent = 'temp/'
    # 确保输出目录存在
    os.makedirs(file_output_parent, exist_ok=True)
    # 处理文件上传
    if files:
        uploaded_files = []
        for file in files:
            if file is not None:
                # 生成唯一的文件名
                file_ext = os.path.splitext(file.filename)[1]
                unique_filename = f'{uuid4()}{file_ext}'
                file_path = os.path.join(file_output_parent, unique_filename)

                # 保存文件
                try:
                    contents = await file.read()
                    with open(file_path, 'wb') as f:
                        f.write(contents)
                    uploaded_files.append({
                        'original_name': file.filename,
                        'saved_name': unique_filename,
                        'path': str(file_path)
                    })
                except Exception as e:
                    logger.error(f"Error saving file {file.filename}: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error saving file {file.filename}")
                finally:
                    await file.close()

        yield SSEMessage(
            event=ChatEventType.STATUS,
            data='文件已收到...'
        ).to_sse()

    # 如果有消息，处理消息（这里先返回一个简单的响应）
    if message:
        logger.info(f'{user.username}: {message}')

        llm = load_gpt4o_mini()

        app = MainAgent(llm=llm).build()
        app.get_graph().draw_mermaid_png(output_file_path='main.png')
        config = {'configurable': {'thread_id': '1'}, 'recursion_limit': 10}
        exclude_events = [
            'on_prompt_start',
            'on_prompt_end',
            'on_parser_start',
            'on_parser_end',
            'on_chat_model_stream',
            'on_chain_stream'
        ]
        async for event in app.astream_events(
                {
                    'messages': [
                        HumanMessage(content=message)],
                    'current_text': current_text
                },
                config,
                version='v2',
                exclude_names=['_write', 'RunnableSequence', 'RunnableLambda']
        ):
            if event['event'] == 'on_chat_model_stream' and event['metadata']['langgraph_node'] == 'router':
                data = event['data']
                if data['chunk'].content:
                    yield SSEMessage(
                        event=ChatEventType.ANSWER,
                        data=data['chunk'].content
                    ).to_sse()

            if event['event'] == 'on_tool_end' and event['name'] == 'modifier':
                modify: OptimizerOutput = event['data']['output']['parsed']
                yield SSEMessage(
                    event=ChatEventType.MODIFY,
                    data=modify.model_dump()
                ).to_sse()

    else:
        yield SSEMessage(
            event=ChatEventType.ANSWER,
            data='文件已收到，请给出你的需求'
        ).to_sse()

    yield SSEMessage(
        event=ChatEventType.STATUS,
        data='消息已收到...'
    ).to_sse()


@router.post('/chat')
async def chat(
        session: SessionDep,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
        project_uid: str = Form(...),
        chat_uid: str = Form(...),
        message: str = Form(None),
        current_text: str = Form(None),
        files: list[UploadFile] = File(None),
):
    logger.debug(f'project: {project_uid} session:{chat_uid}')

    return StreamingResponse(
        event_generator(message, files, current_text, current_user),
        media_type='text/event-stream'
    )
