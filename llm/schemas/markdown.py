from enum import IntEnum

from pydantic import BaseModel


class SourceType(IntEnum):
    MARKDOWN = 0
    PDF = 1
    WEB = 2
    WORD = 3
    POWERPOINT = 4


class FileSource(BaseModel):
    source_url: str = ''
    source_type: int = 0


class MarkdownMeta(BaseModel):
    title: str = ''
    author: str = ''
    year: int
    source: list[FileSource]
