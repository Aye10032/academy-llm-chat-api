from typing import Optional

from sqlmodel import Field

from app.db.session import profile_metadata
from app.schemas.knowledge_base import KnowledgeBase


class KnowledgeBaseTable(KnowledgeBase, table=True, metadata=profile_metadata):
    __tablename__ = 'knowledge_bases'

    id: Optional[int] = Field(default=None, primary_key=True)
