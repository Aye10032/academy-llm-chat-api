from typing import Annotated

from fastapi import Depends
from loguru import logger
from sqlalchemy import MetaData, create_engine
from sqlalchemy_utils import create_database, database_exists
from sqlmodel import Session, SQLModel

from app.core.config import get_settings

if not database_exists(get_settings().base.DATABASE_URL):
    create_database(get_settings().base.DATABASE_URL)
profile_engin = create_engine(get_settings().base.DATABASE_URL)
profile_metadata = MetaData()

if not database_exists(get_settings().knowledge_base.DOC_URL):
    create_database(get_settings().knowledge_base.DOC_URL)
doc_engin = create_engine(get_settings().knowledge_base.DOC_URL)
doc_metadata = MetaData()


@logger.catch
def create_db_and_tables():
    profile_metadata.create_all(profile_engin)


def drop_table(table_name: str):
    metadata = MetaData()
    metadata.reflect(bind=profile_engin)
    table = metadata.tables[table_name]
    if table is not None:
        SQLModel.metadata.drop_all(profile_engin, [table])


def get_session():
    with Session(profile_engin) as session:
        yield session


# 用于单次调用
def get_simple_session():
    return Session(profile_engin)


SessionDep = Annotated[Session, Depends(get_session)]
