import json
from datetime import datetime
from typing import Optional

from langchain_core.documents import Document
from sqlmodel import JSON, Field, SQLModel


class WriteProject(SQLModel):
    uid: str = Field(unique=True)
    user_email: str
    description: str
    last_manuscript: str = ''
    create_time: datetime
    update_time: datetime


class WriteProjectUpdate(SQLModel):
    description: Optional[str] = None
    last_manuscript: Optional[str] = None
    update_time: datetime


class ProjectSources(SQLModel):
    project_uid: str
    sources: str = Field(default='[]', sa_type=JSON)

    def set_sources(self, documents: list[Document]):
        """将 Document 列表序列化为 JSON 字符串"""
        sources_data = []
        for doc in documents:
            sources_data.append({'page_content': doc.page_content, 'metadata': doc.metadata})
        self.sources = json.dumps(sources_data)

    def get_sources(self) -> list[Document]:
        """将 JSON 字符串反序列化为 Document 列表"""
        sources_data = json.loads(self.sources)
        return [
            Document(page_content=item['page_content'], metadata=item['metadata'])
            for item in sources_data
        ]


class ChatRecord(SQLModel):
    project_uid: str
    user_email: str
    price: float
    create_time: datetime
