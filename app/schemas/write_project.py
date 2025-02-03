from datetime import datetime

from sqlmodel import SQLModel, Field


class WriteProject(SQLModel):
    uid: str = Field(unique=True)
    graph_checkpoint: str
    user_email: str
    description: str
    last_manuscript: str = ''
    create_time: datetime
    update_time: datetime
