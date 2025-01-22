from typing import Optional

from sqlmodel import select, Session

from app.models import ChatSessionTable


def get_chat_list(
        session: Session,
        email: str,
        knowledge_base_name: str
) -> Optional[ChatSessionTable]:
    statement = (select(ChatSessionTable)
                 .where(ChatSessionTable.user_email == email)
                 .where(ChatSessionTable.knowledge_base_name == knowledge_base_name))
    return session.exec(statement).all()


def get_chat(session: Session, email: str, history_id: str) -> Optional[ChatSessionTable]:
    statement = (select(ChatSessionTable)
                 .where(ChatSessionTable.user_email == email)
                 .where(ChatSessionTable.history_id == history_id))
    return session.exec(statement).first()


def insert_chat(session: Session, chat_session: ChatSessionTable) -> None:
    session.add(chat_session)
    session.commit()
