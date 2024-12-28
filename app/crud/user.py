from typing import Optional

from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.db.session import SessionDep
from app.models import UserTable
from app.schemas.user import UserPublic


class UserExistError(ValueError):
    pass


def get_user(session: SessionDep, email: str) -> Optional[UserTable]:
    statement = select(UserTable).where(UserTable.email == email)
    return session.exec(statement).first()


def insert_user(session: SessionDep, user: UserTable) -> None:
    try:
        logger.debug(f'新增用户请求 ({UserPublic.model_validate(user)})')
        session.add(user)
        session.commit()
    except IntegrityError as e:
        if 'UNIQUE' in e.args[0]:
            raise UserExistError from e
        else:
            raise e
