from datetime import datetime

from sqlmodel import SQLModel


class ChatSession(SQLModel):
    chat_uid: str
    parent_uid: str
    user_email: str
    description: str = '新建对话'
    create_time: datetime
    update_time: datetime
