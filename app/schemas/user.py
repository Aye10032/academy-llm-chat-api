from enum import IntEnum

from sqlmodel import SQLModel, Field


class UserRole(IntEnum):
    visitor = 0
    writer = 1
    admin = 2


class UserBase(SQLModel):
    email: str = Field(index=True, unique=True)
    is_active: bool = Field(default=True)
    role: UserRole = Field(default=UserRole.visitor)

class UserPublic(UserBase):
    username: str = Field(default="")
