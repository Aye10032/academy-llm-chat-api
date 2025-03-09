from enum import IntEnum
from typing import Optional

from sqlmodel import Field, SQLModel


class ManuscriptType(IntEnum):
    CONTEXT = 0
    DRAFT = 1
    LOCAL_FILE = 2


class Manuscript(SQLModel):
    uid: str
    project_uid: str
    title: str
    content: str = ''
    version: int = 0
    file_type: ManuscriptType = Field(default=ManuscriptType.CONTEXT)


class ManuscriptPublic(SQLModel):
    uid: str
    project_uid: str
    title: str
    version: int
    file_type: ManuscriptType = Field(default=ManuscriptType.CONTEXT)


class ManuscriptUpdate(SQLModel):
    title: Optional[str] = None
    content: Optional[str] = None
    version: Optional[int] = None
