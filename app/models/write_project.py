from typing import Optional

from sqlmodel import Field

from app.schemas.write_project import WriteProject


class WriteProjectTable(WriteProject, table=True):
    __tablename__ = "write_project"

    id: Optional[int] = Field(default=None, primary_key=True)
