from typing import Optional

from sqlmodel import Field

from app.schemas.user import User


class UserTable(User, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str
    is_active: bool = Field(default=True)
