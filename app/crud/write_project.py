from sqlmodel import Session

from app.models import WriteProjectTable


def insert_project(session: Session, project: WriteProjectTable) -> None:
    session.add(project)
    session.commit()