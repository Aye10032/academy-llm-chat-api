from datetime import datetime

from sqlmodel import SQLModel


class ChatSession(SQLModel):
    history_id: str
    knowledge_base_name: str
    user_email: str
    description: str = '新建对话'
    create_time: datetime
    update_time: datetime
