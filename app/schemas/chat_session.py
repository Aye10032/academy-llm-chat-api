from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel


class ChatSession(SQLModel):
    uid: str
    parent_uid: str
    user_email: str
    description: str = '新建对话'
    create_time: datetime
    update_time: datetime


class ChatSessionUpdate(SQLModel):
    description: Optional[str] = None
    update_time: Optional[datetime] = None
