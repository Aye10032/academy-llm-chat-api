from typing import Optional

from sqlmodel import Session, select

from app.models import WriteProjectTable


def get_project_list(
        session: Session,
        email: str,
) -> Optional[WriteProjectTable]:
    statement = select(WriteProjectTable).where(WriteProjectTable.user_email == email)
    return session.exec(statement).all()


def insert_project(session: Session, project: WriteProjectTable) -> WriteProjectTable:
    session.add(project)
    session.commit()
    session.refresh(project)
    return project
