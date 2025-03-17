from typing import Optional

from sqlmodel import Field

from app.db.session import profile_metadata
from app.schemas.write_project import ChatRecord, ProjectSources, WriteProject


class WriteProjectTable(WriteProject, table=True):
    __tablename__ = 'write_project'
    metadata = profile_metadata

    id: Optional[int] = Field(default=None, primary_key=True)


class ProjectSourcesTable(ProjectSources, table=True):
    __tablename__ = 'project_source'
    metadata = profile_metadata

    id: Optional[int] = Field(default=None, primary_key=True)


class ChatRecordTable(ChatRecord, table=True):
    __tablename__ = 'chat_record'
    metadata = profile_metadata

    id: Optional[int] = Field(default=None, primary_key=True)
