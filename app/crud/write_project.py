from typing import Optional

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models import WriteProjectTable
from app.schemas.write_project import WriteProjectUpdate


def get_project_list(
        session: Session,
        email: str,
) -> list[WriteProjectTable]:
    statement = select(WriteProjectTable).where(WriteProjectTable.user_email == email)
    return session.exec(statement).all()


def get_project(session, uid) -> Optional[WriteProjectTable]:
    statement = select(WriteProjectTable).where(WriteProjectTable.uid == uid)
    return session.exec(statement).first()


def insert_project(session: Session, project: WriteProjectTable) -> WriteProjectTable:
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


def update_project(
        session: Session, project_uid: str, project: WriteProjectUpdate
) -> WriteProjectTable:
    db_project = get_project(session, project_uid)
    if not db_project:
        raise HTTPException(status_code=404, detail='该记录不存在！')

    project_data = project.model_dump(exclude_unset=True)
    db_project.sqlmodel_update(project_data)
    session.add(db_project)
    session.commit()
    session.refresh(db_project)
    return db_project
