from typing import Optional

from fastapi import HTTPException
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlmodel import select, Session

from app.models import UserTable
from app.schemas.user import UserPublic, UserUpdate


class UserExistError(ValueError):
    pass


def insert(session: Session, user: UserTable) -> None:
    try:
        logger.debug(f'新增用户请求 ({UserPublic.model_validate(user)})')
        session.add(user)
        session.commit()
    except IntegrityError as e:
        if 'UNIQUE' in e.args[0]:
            raise UserExistError from e
        else:
            raise e


def update(session: Session, email: str, user: UserUpdate) -> UserTable:
    db_user = get(session, email)
    if not db_user:
        raise HTTPException(status_code=404, detail='用户不存在！')

    user_data = user.model_dump(exclude_unset=True)
    db_user.sqlmodel_update(user_data)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


def get(session: Session, email: str) -> Optional[UserTable]:
    statement = select(UserTable).where(UserTable.email == email)
    return session.exec(statement).first()
