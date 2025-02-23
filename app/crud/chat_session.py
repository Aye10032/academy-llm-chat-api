from typing import Optional

from fastapi import HTTPException
from sqlmodel import select, Session

from app.models import ChatSessionTable
from app.schemas.chat_session import ChatSessionUpdate


def insert(session: Session, chat_session: ChatSessionTable) -> ChatSessionTable:
    session.add(chat_session)
    session.commit()
    session.refresh(chat_session)
    return chat_session


def delete(session: Session, uid: str):
    statement = select(ChatSessionTable).where(ChatSessionTable.uid == uid)
    results = session.exec(statement)
    chat_session = results.one()
    session.delete(chat_session)
    session.commit()


def delete_by_parent(session: Session, parent_uid: str):
    statement = select(ChatSessionTable).where(
        ChatSessionTable.parent_uid == parent_uid
    )
    results = session.exec(statement)
    chat_session = results.all()
    session.delete(chat_session)
    session.commit()


def update(
    session: Session, chat_uid: str, chat_session: ChatSessionUpdate
) -> ChatSessionTable:
    db_chat = get(session, chat_uid)
    if not db_chat:
        raise HTTPException(status_code=404, detail='该记录不存在！')

    chat_data = chat_session.model_dump(exclude_unset=True)
    db_chat.sqlmodel_update(chat_data)
    session.add(db_chat)
    session.commit()
    session.refresh(db_chat)
    return db_chat


def get_list(
    session: Session, parent_uid: str, use_email: str
) -> Optional[ChatSessionTable]:
    statement = select(ChatSessionTable).where(
        ChatSessionTable.parent_uid == parent_uid
    ).where(ChatSessionTable.user_email == use_email)
    return session.exec(statement).all()


def get(session: Session, chat_uid: str) -> Optional[ChatSessionTable]:
    statement = select(ChatSessionTable).where(ChatSessionTable.uid == chat_uid)
    return session.exec(statement).first()
