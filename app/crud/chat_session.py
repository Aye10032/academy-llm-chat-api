from typing import Optional

from sqlmodel import select

from app.db.session import SessionDep
from app.models import ChatSessionTable


def get_chat_list(
        session: SessionDep,
        email: str,
        knowledge_base_name: str
) -> Optional[ChatSessionTable]:
    statement = (select(ChatSessionTable)
                 .where(ChatSessionTable.user_email == email)
                 .where(ChatSessionTable.knowledge_base_name == knowledge_base_name))
    return session.exec(statement).all()


def get_chat(session: SessionDep, chat_history: str) -> Optional[ChatSessionTable]:
    statement = select(ChatSessionTable).where(ChatSessionTable.chat_history == chat_history)
    return session.exec(statement).first()


def insert_chat(session: SessionDep, chat_session: ChatSessionTable):
    session.add(chat_session)
    session.commit()
