from enum import IntEnum

from pydantic import EmailStr
from sqlmodel import SQLModel, Field


class UserRole(IntEnum):
    VISITOR = 0
    WRITER = 1
    ADMIN = 2


class User(SQLModel):
    username: str = Field(default="")
    email: EmailStr = Field(index=True, unique=True)
    role: UserRole = Field(default=UserRole.VISITOR)


class UserPublic(User):
    is_active: bool = Field(default=True)
