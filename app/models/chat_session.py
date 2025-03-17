from typing import Optional

from sqlmodel import Field

from app.db.session import profile_metadata
from app.schemas.chat_session import ChatSession


class ChatSessionTable(ChatSession, table=True, metadata=profile_metadata):
    __tablename__ = 'chat_session'

    id: Optional[int] = Field(default=None, primary_key=True)
