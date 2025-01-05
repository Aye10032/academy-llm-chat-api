from typing import Optional

from fastapi import HTTPException
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.db.session import SessionDep
from app.models import UserTable
from app.schemas.user import UserPublic, UserUpdate


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


def update_user(session: SessionDep, email: str, user: UserUpdate) -> UserTable:
    db_user = get_user(session, email)
    if not db_user:
        raise HTTPException(status_code=404, detail='用户不存在！')

    user_data = user.model_dump(exclude_unset=True)
    db_user.sqlmodel_update(user_data)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user
