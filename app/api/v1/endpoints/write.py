from datetime import datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Query
from loguru import logger
from pydantic import BaseModel

from app.core.security import get_current_active_user
from app.crud.chat_session import insert_chat, get_chat_list
from app.crud.write_project import insert_project, get_project_list
from app.db.session import SessionDep
from app.models import UserTable, WriteProjectTable, ChatSessionTable
from app.schemas.chat_session import ChatSession
from app.schemas.write_project import WriteProject

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    project_uid: str
    chat_uid: str


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


@router.patch('/new_chat', description='新建写作工程')
async def add_new_chat(
        session: SessionDep,
        project_uid: str,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
) -> str:
    now_time = datetime.now()

    new_chat = ChatSessionTable(
        chat_uid=str(uuid4()),
        parent_uid=project_uid,
        description='',
        user_email=str(current_user.email),
        create_time=now_time,
        update_time=now_time
    )

    new_chat = insert_chat(session, new_chat)
    return new_chat.chat_uid


@router.post('/chat')
async def chat(
        session: SessionDep,
        request: ChatRequest,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    logger.debug(f'project: {request.project_uid} session:{request.chat_uid}')
    logger.info(f'{current_user.username}: {request.message}')

    file_output_parent = 'temp/'
