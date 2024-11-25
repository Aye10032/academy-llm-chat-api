from typing import Optional

from sqlmodel import Field

from app.schemas.user import UserBase


class User(UserBase, table=True):
    __tablename__ = "users"

    id: int = Field(default=None, primary_key=True)
    username: str = Field(default="")
    hashed_password: str
