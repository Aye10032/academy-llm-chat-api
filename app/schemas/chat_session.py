from datetime import datetime

from sqlmodel import SQLModel


class ChatSession(SQLModel):
    chat_history: str
    knowledge_base_name: str
    user_email: str
    description: str = '新建对话'
    create_time: datetime
    update_time: datetime
