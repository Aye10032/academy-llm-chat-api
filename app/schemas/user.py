from enum import IntEnum

from sqlmodel import SQLModel, Field


class UserRole(IntEnum):
    VISITOR = 0
    WRITER = 1
    ADMIN = 2


class UserPublic(SQLModel):
    email: str = Field(index=True, unique=True)
    username: str = Field(default="")
    is_active: bool = Field(default=True)
    role: UserRole = Field(default=UserRole.VISITOR)
