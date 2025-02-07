from typing import Optional

from sqlmodel import Field

from app.schemas.manuscript import Manuscript


class ManuscriptTable(Manuscript, table=True):
    __tablename__ = 'manuscript'

    id: Optional[int] = Field(default=None, primary_key=True)
