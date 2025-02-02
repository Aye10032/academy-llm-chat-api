from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, model_validator
from sqlmodel import SQLModel


class WriteProject(SQLModel):
    history_id: str
    graph_checkpoint: str
    user_email: str
    description: str = '新建工程'
    create_time: datetime
    update_time: datetime


class FileStructure(BaseModel):
    file_id: str
    name: str
    type: Literal['file', 'folder']
    children: Optional[list['FileStructure']] = None

    @model_validator(mode='after')
    def verify_file_type(self):
        if self.type == 'file':
            assert self.children is None

        return self
