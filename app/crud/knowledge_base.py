from fastapi import HTTPException
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlmodel import select, Session

from app.models import KnowledgeBaseTable
from app.schemas.knowledge_base import KnowledgeBaseUpdate


class KBExistError(ValueError):
    pass


def get_knowledge_base(session: Session, table_name: str):
    statement = select(KnowledgeBaseTable).where(KnowledgeBaseTable.table_name == table_name)
    return session.exec(statement).first()


def get_knowledge_bases(session: Session, offset: int, limit: int):
    statement = select(KnowledgeBaseTable).offset(offset).limit(limit)
    return session.exec(statement).all()


def insert_knowledge_base(session: Session, knowledge_base: KnowledgeBaseTable):
    try:
        session.add(knowledge_base)
        session.commit()
    except IntegrityError as e:
        if 'UNIQUE' in e.args[0]:
            raise KBExistError from e
        else:
            raise e

    logger.info(f'新增数据 {KnowledgeBaseTable.__tablename__}:{knowledge_base.table_name}')


def update_knowledge_base(
        session: Session, table_name: str, kb: KnowledgeBaseUpdate
) -> KnowledgeBaseTable:
    db_kb = get_knowledge_base(session, table_name)
    if not db_kb:
        raise HTTPException(status_code=404, detail='用户不存在！')

    kb_data = kb.model_dump(exclude_unset=True)
    db_kb.sqlmodel_update(kb_data)
    session.add(db_kb)
    session.commit()
    session.refresh(db_kb)
    return db_kb


def delete_knowledge_base(session: Session, table_name: str):
    statement = select(KnowledgeBaseTable).where(KnowledgeBaseTable.table_name == table_name)
    results = session.exec(statement)
    kb = results.first()
    if kb:
        logger.info(f'删除数据 {KnowledgeBaseTable.__tablename__}:{table_name}')
        session.delete(kb)
        session.commit()
