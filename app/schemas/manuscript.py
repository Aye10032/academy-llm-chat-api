from typing import Optional

from sqlmodel import SQLModel


class Manuscript(SQLModel):
    uid: str
    project_uid: str
    title: str
    content: str = ''
    version: int = 0
    is_draft: bool = False

class ManuscriptPublic(SQLModel):
    uid: str
    project_uid: str
    title: str
    version: int
    is_draft: bool = False

class ManuscriptUpdate(SQLModel):
    title: Optional[str] = None
    content: Optional[str] = None
    version:Optional[int] = None
