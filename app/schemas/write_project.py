from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field


class WriteProject(SQLModel):
    uid: str = Field(unique=True)
    user_email: str
    description: str
    last_manuscript: str = ''
    create_time: datetime
    update_time: datetime


class WriteProjectUpdate(SQLModel):
    description: Optional[str] = None
    last_manuscript: Optional[str] = None
    update_time: datetime
