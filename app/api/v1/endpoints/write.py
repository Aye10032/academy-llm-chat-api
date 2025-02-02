from datetime import datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends

from app.core.security import get_current_active_user
from app.crud.write_project import insert_project
from app.db.session import SessionDep
from app.models import UserTable, WriteProjectTable

router = APIRouter()


@router.patch('/write/new', description='新建写作工程')
async def add_new_project(
        session: SessionDep,
        current_user: Annotated[UserTable, Depends(get_current_active_user)]
) -> str:
    now_time = datetime.now()

    new_project = WriteProjectTable(
        history_id=str(uuid4()),
        graph_checkpoint=str(uuid4()),
        user_email=str(current_user.email),
        create_time=now_time,
        update_time=now_time
    )

    insert_project(session, new_project)
    return new_project.history_id
