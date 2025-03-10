from typing import Annotated

from fastapi import Depends
from loguru import logger
from sqlalchemy import MetaData, create_engine
from sqlmodel import Session, SQLModel

from app.core.config import get_settings

connect_args = {'check_same_thread': False}
engine = create_engine(get_settings().base.DATABASE_URL, connect_args=connect_args)


@logger.catch
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def drop_table(table_name: str):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables[table_name]
    if table is not None:
        SQLModel.metadata.drop_all(engine, [table])


def get_session():
    with Session(engine) as session:
        yield session


# 用于单次调用
def get_simple_session():
    return Session(engine)


SessionDep = Annotated[Session, Depends(get_session)]
