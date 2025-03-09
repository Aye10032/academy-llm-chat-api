from typing import Optional

from fastapi import HTTPException
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.models import KnowledgeBaseTable
from app.schemas.knowledge_base import KnowledgeBaseUpdate


class KBExistError(ValueError):
    pass


def insert(session: Session, knowledge_base: KnowledgeBaseTable) -> None:
    try:
        session.add(knowledge_base)
        session.commit()
    except IntegrityError as e:
        if 'UNIQUE' in e.args[0]:
            raise KBExistError from e
        else:
            raise e


def delete(session: Session, uid: str):
    statement = select(KnowledgeBaseTable).where(KnowledgeBaseTable.uid == uid)
    results = session.exec(statement)
    kb = results.first()
    if kb:
        logger.info(f'删除数据 {KnowledgeBaseTable.__tablename__}:{kb.table_name}')
        session.delete(kb)
        session.commit()


def delete_by_name(session: Session, name: str):
    statement = select(KnowledgeBaseTable).where(KnowledgeBaseTable.table_name == name)
    results = session.exec(statement)
    kb = results.first()
    if kb:
        logger.info(f'删除数据 {KnowledgeBaseTable.__tablename__}:{kb.table_name}')
        session.delete(kb)
        session.commit()


def update(session: Session, uid: str, kb: KnowledgeBaseUpdate) -> KnowledgeBaseTable:
    db_kb = get(session, uid)
    if not db_kb:
        raise HTTPException(status_code=404, detail='用户不存在！')

    kb_data = kb.model_dump(exclude_unset=True)
    db_kb.sqlmodel_update(kb_data)
    session.add(db_kb)
    session.commit()
    session.refresh(db_kb)
    return db_kb


def get_list(session: Session, offset: int, limit: int) -> list[KnowledgeBaseTable]:
    statement = select(KnowledgeBaseTable).offset(offset).limit(limit)
    return session.exec(statement).all()


def get_by_name(session: Session, name: str) -> Optional[KnowledgeBaseTable]:
    statement = select(KnowledgeBaseTable).where(KnowledgeBaseTable.table_name == name)
    return session.exec(statement).first()


def get(session: Session, uid: str) -> Optional[KnowledgeBaseTable]:
    statement = select(KnowledgeBaseTable).where(KnowledgeBaseTable.uid == uid)
    return session.exec(statement).first()
