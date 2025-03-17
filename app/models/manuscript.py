from typing import Optional

from sqlmodel import Field

from app.db.session import profile_metadata
from app.schemas.manuscript import Manuscript


class ManuscriptTable(Manuscript, table=True):
    __tablename__ = 'manuscript'
    metadata = profile_metadata

    id: Optional[int] = Field(default=None, primary_key=True)
