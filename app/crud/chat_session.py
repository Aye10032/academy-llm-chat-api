from typing import Optional

from sqlmodel import select, Session

from app.models import ChatSessionTable


def get_chat_list(
        session: Session,
        email: str,
        parent_uid: str
) -> Optional[ChatSessionTable]:
    statement = (select(ChatSessionTable)
                 .where(ChatSessionTable.user_email == email)
                 .where(ChatSessionTable.parent_uid == parent_uid))
    return session.exec(statement).all()


def get_chat(session: Session, email: str, chat_uid: str) -> Optional[ChatSessionTable]:
    statement = (select(ChatSessionTable)
                 .where(ChatSessionTable.user_email == email)
                 .where(ChatSessionTable.chat_uid == chat_uid))
    return session.exec(statement).first()


def insert_chat(session: Session, chat_session: ChatSessionTable) -> None:
    session.add(chat_session)
    session.commit()
