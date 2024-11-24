from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import jwt
from typing import Optional

from app.core.config import settings
from app.models.user import User
from app.db.session import SessionLocal

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


def get_db():
    """
    获取数据库会话的依赖项
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db)
) -> User:
    """
    获取当前认证用户的依赖项
    
    Args:
        token: JWT token
        db: 数据库会话
    
    Returns:
        User: 当前认证用户
        
    Raises:
        HTTPException: 认证失败时抛出 401 异常
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user: Optional[User] = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
        current_user: User = Depends(get_current_user)
) -> User:
    """
    获取当前活跃用户的依赖项
    
    Args:
        current_user: 当前认证用户
        
    Returns:
        User: 当前活跃用户
        
    Raises:
        HTTPException: 用户未激活时抛出 400 异常
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="用户未激活")
    return current_user
