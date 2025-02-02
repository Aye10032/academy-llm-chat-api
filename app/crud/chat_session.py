from typing import Optional

from fastapi import HTTPException
from sqlmodel import select, Session

from app.models import ChatSessionTable
from app.schemas.chat_session import ChatSessionUpdate


def get_chat_list(
        session: Session,
        email: str,
        parent_uid: str
) -> Optional[ChatSessionTable]:
    statement = (select(ChatSessionTable)
                 .where(ChatSessionTable.user_email == email)
                 .where(ChatSessionTable.parent_uid == parent_uid))
    return session.exec(statement).all()


def get_chat(session: Session, chat_uid: str) -> Optional[ChatSessionTable]:
    statement = (select(ChatSessionTable)
                 .where(ChatSessionTable.chat_uid == chat_uid))
    return session.exec(statement).first()


def insert_chat(session: Session, chat_session: ChatSessionTable) -> None:
    session.add(chat_session)
    session.commit()


def update_chat(
        session: Session, chat_uid: str, chat_session: ChatSessionUpdate
) -> ChatSessionTable:
    db_chat = get_chat(session, chat_uid)
    if not db_chat:
        raise HTTPException(status_code=404, detail='该记录不存在！')

    chat_data = chat_session.model_dump(exclude_unset=True)
    db_chat.sqlmodel_update(chat_data)
    session.add(db_chat)
    session.commit()
    session.refresh(db_chat)
    return db_chat
