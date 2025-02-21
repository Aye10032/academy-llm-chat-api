from typing import Optional

from sqlmodel import Field

from app.schemas.write_project import WriteProject, ProjectSources, ChatRecord


class WriteProjectTable(WriteProject, table=True):
    __tablename__ = 'write_project'

    id: Optional[int] = Field(default=None, primary_key=True)


class ProjectSourcesTable(ProjectSources, table=True):
    __tablename__ = 'project_source'

    id: Optional[int] = Field(default=None, primary_key=True)


class ChatRecordTable(ChatRecord, table=True):
    __tablename__ = 'chat_record'

    id: Optional[int] = Field(default=None, primary_key=True)
