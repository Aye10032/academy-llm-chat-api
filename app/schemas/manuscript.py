from typing import Optional

from sqlmodel import SQLModel


class Manuscript(SQLModel):
    uid: str
    project_uid: str
    title: str
    content: str
    version: int
    is_draft: bool = False


class ManuscriptUpdate(SQLModel):
    title: Optional[str] = None
    content: Optional[str] = None
    version: Optional[str] = None
