from typing import Optional

from fastapi import HTTPException
from sqlalchemy import func
from sqlmodel import Session, select

from app.models import ManuscriptTable
from app.schemas.manuscript import ManuscriptUpdate


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
                 .where(ManuscriptTable.uid == uid)
                 .order_by(ManuscriptTable.version.desc())
                 .limit(1))
    return session.exec(statement).first()


def insert_manuscript(
        session: Session, manuscript: ManuscriptTable
) -> ManuscriptTable:
    session.add(manuscript)
    session.commit()
    session.refresh(manuscript)
    return manuscript


def update_manuscript(
        session: Session, uid: str, manuscript: ManuscriptUpdate
) -> ManuscriptTable:
    db_manuscript = get_manuscript(session, uid)
    if not db_manuscript:
        raise HTTPException(status_code=404, detail='该记录不存在！')

    last_version = db_manuscript.version
    manuscript.version = last_version + 1
    update_data = manuscript.model_dump(exclude_unset=True)

    db_manuscript.sqlmodel_update(update_data)
    session.add(db_manuscript)
    session.commit()
    session.refresh(db_manuscript)
    return db_manuscript
