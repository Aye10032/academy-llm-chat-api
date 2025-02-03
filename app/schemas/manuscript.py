from sqlmodel import SQLModel


class Manuscript(SQLModel):
    uid: str
    name: str
    project_uid: str
    content: str
    version: int
