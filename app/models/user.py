from enum import IntEnum
from typing import Optional
from pydantic import BaseModel, EmailStr, ConfigDict
from sqlalchemy import Boolean, Column, Integer, String, Enum
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class UserRole(IntEnum):
    visitor = 0
    writer = 1
    admin = 2


# SQLAlchemy Model
class User(Base):
    __tablename__ = "users"

    id: int = Column(Integer, primary_key=True, index=True)
    nick_name: str = Column(String, default='')
    email: str = Column(String, unique=True, index=True)
    hashed_password: str = Column(String)
    is_active: bool = Column(Boolean, default=True)
    role: UserRole = Column(Enum(UserRole), default=UserRole.visitor)


# Pydantic Models
class UserBase(BaseModel):
    """用于基本用户信息的共享属性"""
    email: EmailStr
    nick_name: str = ""
    is_active: bool = True
    role: UserRole = UserRole.visitor


class UserCreate(UserBase):
    """用于创建用户时的请求体"""
    password: str


class UserUpdate(BaseModel):
    """用于更新用户时的请求体"""
    email: Optional[EmailStr] = None
    nick_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[UserRole] = None


class UserInDB(UserBase):
    """用于数据库操作的完整用户模型"""
    id: int
    hashed_password: str

    model_config = ConfigDict(from_attributes=True)


class UserResponse(UserBase):
    """用于API响应的用户模型"""
    id: int

    model_config = ConfigDict(from_attributes=True)
