from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class KnowledgeBase(SQLModel):
    uid: str = Field(unique=True)
    table_name: str = Field(unique=True)
    table_title: str = ''
    description: str = ''
    create_time: datetime
    last_update: datetime
    is_public: bool = True
    is_active: bool = True


class KnowledgeBaseUpdate(SQLModel):
    table_title: Optional[str] = None
    description: Optional[str] = None
    last_update: Optional[datetime] = None
    is_public: bool = True
    is_active: Optional[bool] = None
