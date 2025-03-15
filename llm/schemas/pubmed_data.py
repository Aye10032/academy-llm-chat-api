from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None


class Journal(BaseModel):
    name: str
    issn: Optional[str] = None


class PubMedData(BaseModel):
    pmid: str
    title: str
    pub_date: date
    author: list[Author] = Field(default_factory=list)
    journal: Optional[Journal] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    reference_num: int = 0
    vector_db_uid: Optional[str] = None
