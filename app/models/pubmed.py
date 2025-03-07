from typing import Optional

from sqlmodel import Field

from app.schemas.pubmed import PubMedPaper, PubMedReference


class PubMedPaperTable(PubMedPaper, table=True):
    __tablename__ = 'pubmed_paper'

    id: Optional[int] = Field(default=None, primary_key=True)


class PubMedReferenceTable(PubMedReference, table=True):
    __tablename__ = 'pubmed_reference'

    id: Optional[int] = Field(default=None, primary_key=True)
