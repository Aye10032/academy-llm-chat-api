from typing import Any, Dict, Optional, Union
from sqlalchemy.orm import Session
from pydantic import EmailStr

from app.core.security import get_password_hash
from app.models.user import User, UserCreate, UserUpdate


class CRUDUser:
    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100):
        return db.query(User).offset(skip).limit(limit).all()

    @staticmethod
    def create(db: Session, user_in: UserCreate) -> User:
        db_user = User(
            email=user_in.email,
            hashed_password=get_password_hash(user_in.password),
            nick_name=user_in.nick_name,
            role=user_in.role,
            is_active=user_in.is_active
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def update(
            db: Session,
            db_user: User,
            user_in: Union[UserUpdate, Dict[str, Any]]
    ) -> User:
        if isinstance(user_in, dict):
            update_data = user_in
        else:
            update_data = user_in.model_dump(exclude_unset=True)

        if update_data.get("password"):
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password

        for field, value in update_data.items():
            setattr(db_user, field, value)

        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def delete_by_email(db: Session, email: str) -> Optional[User]:
        user = db.query(User).filter(User.email == email).first()
        if user:
            db.delete(user)
            db.commit()
        return user


# 创建实例供其他模块使用
crud_user = CRUDUser()
