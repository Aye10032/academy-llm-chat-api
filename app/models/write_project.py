from typing import Optional

from sqlmodel import Field

from app.db.session import profile_metadata
from app.schemas.write_project import ChatRecord, ProjectSources, WriteProject


class WriteProjectTable(WriteProject, table=True, metadata=profile_metadata):
    __tablename__ = 'write_project'

    id: Optional[int] = Field(default=None, primary_key=True)


class ProjectSourcesTable(ProjectSources, table=True, metadata=profile_metadata):
    __tablename__ = 'project_source'

    id: Optional[int] = Field(default=None, primary_key=True)


class ChatRecordTable(ChatRecord, table=True, metadata=profile_metadata):
    __tablename__ = 'chat_record'

    id: Optional[int] = Field(default=None, primary_key=True)
