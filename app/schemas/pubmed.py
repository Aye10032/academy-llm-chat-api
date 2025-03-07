from sqlmodel import Field, SQLModel


class PubMedPaper(SQLModel):
    pmid: str = Field(unique=True)
    title: str
    year: int
    author: str
    journal: str = ''
    doi: str = ''
    abstract: str = ''
    has_paper: bool = False
    paper_uid: str = ''


class PubMedReference(SQLModel):
    origin: str
    cite: str
