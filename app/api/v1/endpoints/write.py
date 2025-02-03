from datetime import datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends
from loguru import logger
from pydantic import BaseModel

from app.core.security import get_current_active_user
from app.crud.write_project import insert_project
from app.db.session import SessionDep
from app.models import UserTable, WriteProjectTable

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    project_uid: str
    chat_uid: str


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


@router.post('/chat')
async def chat(
        session: SessionDep,
        request: ChatRequest,
        current_user: Annotated[UserTable, Depends(get_current_active_user)],
):
    logger.debug(f'knowledge_base: {request.knowledge_base_name} session:{request.history_id}')
    logger.info(f'{current_user.username}: {request.message}')
