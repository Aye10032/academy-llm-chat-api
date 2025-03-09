from enum import IntEnum
from typing import Optional

from pydantic import EmailStr
from sqlmodel import Field, SQLModel


class UserRole(IntEnum):
    VISITOR = 0
    WRITER = 1
    ADMIN = 2


class User(SQLModel):
    username: str = Field(default='')
    email: EmailStr = Field(index=True, unique=True)
    role: UserRole = Field(default=UserRole.VISITOR)
    last_knowledge_base: str = Field(default='')
    last_project: str = Field(default='')


class UserPublic(User):
    is_active: bool = Field(default=True)


class UserUpdate(SQLModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None
    last_knowledge_base: Optional[str] = None
    last_project: Optional[str] = None
    is_active: Optional[bool] = None
