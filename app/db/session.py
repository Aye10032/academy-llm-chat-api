from typing import Annotated

from fastapi import Depends
from loguru import logger
from sqlalchemy import create_engine
from sqlmodel import SQLModel, Session

from app.core.config import settings

connect_args = {"check_same_thread": False}
engine = create_engine(settings.SQLITE_DATABASE_URL, connect_args=connect_args)


def create_db_and_tables():
    logger.debug('create db')
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
