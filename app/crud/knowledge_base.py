from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.db.session import SessionDep
from app.models import KnowledgeBaseTable
from app.schemas.knowledge_base import KnowledgeBase


class KBExistError(ValueError):
    pass


def get_knowledge_base(session: SessionDep, table_name: str) -> KnowledgeBaseTable:
    statement = select(KnowledgeBaseTable).where(KnowledgeBaseTable.table_name == table_name)
    return session.exec(statement).first()


def get_knowledge_bases(session: SessionDep, offset: int, limit: int) -> list[KnowledgeBaseTable]:
    statement = select(KnowledgeBaseTable).offset(offset).limit(limit)
    return session.exec(statement).all()


def insert_knowledge_base(session: SessionDep, knowledge_base: KnowledgeBaseTable):
    try:
        logger.debug(f'新增用户请求 ({KnowledgeBase.model_validate(knowledge_base)})')
        session.add(knowledge_base)
        session.commit()
    except IntegrityError as e:
        if 'UNIQUE' in e.args[0]:
            raise KBExistError from e
        else:
            raise e
