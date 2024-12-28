from datetime import datetime

from sqlmodel import SQLModel, Field


class KnowledgeBase(SQLModel):
    table_name: str = Field(unique=True)
    table_title: str = ''
    description: str = ''
    create_time: datetime
    last_update: datetime
    is_active: bool = Field(default=True)
