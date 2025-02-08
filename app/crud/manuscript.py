from typing import Optional

from sqlalchemy import func
from sqlmodel import Session, select

from app.models import ManuscriptTable


def get_manuscripts_list(session: Session, project_uid: str):
    subquery = (
        select(
            ManuscriptTable.uid,
            func.max(ManuscriptTable.version).label("max_version")
        )
        .where(ManuscriptTable.project_uid == project_uid)
        .group_by(ManuscriptTable.uid)
        .subquery()
    )

    statement = (
        select(ManuscriptTable)
        .where(ManuscriptTable.project_uid == project_uid)
        .join(
            subquery,
            (ManuscriptTable.uid == subquery.c.uid) & (ManuscriptTable.version == subquery.c.max_version)
        )
    )

    result = session.exec(statement)
    return result.all()


def get_manuscript(session: Session, uid: str) -> Optional[ManuscriptTable]:
    statement = (select(ManuscriptTable)
                 .where(ManuscriptTable.uid == uid))
    return session.exec(statement).first()


def insert_manuscript(
        session: Session, manuscript: ManuscriptTable
) -> ManuscriptTable:
    session.add(manuscript)
    session.commit()
    session.refresh(manuscript)
    return manuscript
