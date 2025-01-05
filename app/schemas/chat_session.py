from datetime import datetime

from pydantic import UUID4
from sqlmodel import SQLModel


class ChatSession(SQLModel):
    chat_history: UUID4
    knowledge_base_name: str
    user_email: str
    description: str = ''
    create: datetime
    last_update: datetime
