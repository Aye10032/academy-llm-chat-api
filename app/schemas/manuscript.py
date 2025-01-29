from sqlmodel import SQLModel


class Manuscript(SQLModel):
    name: str
    project: str
    version: int
    is_draft: bool = False
