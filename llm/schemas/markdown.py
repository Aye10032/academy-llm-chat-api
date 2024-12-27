from pydantic import BaseModel

MARKDOWN = 0
PDF = 1
WEB = 2
WORD = 3
POWERPOINT = 4


class MarkdownMeta(BaseModel):
    title: str = ''
    author: str = ''
    year: int
    source: str = ''
    source_type: int = MARKDOWN
