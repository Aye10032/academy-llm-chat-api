from typing import Optional

from sqlmodel import Field

from app.db.session import profile_metadata
from app.schemas.user import User


class UserTable(User, table=True, metadata=profile_metadata):
    __tablename__ = 'users'

    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str
    is_active: bool = Field(default=True)
