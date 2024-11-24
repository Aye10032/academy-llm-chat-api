from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import create_access_token
from app.services.auth import authenticate_user
from app.api.deps import get_current_user, get_db
from app.models.user import User

router = APIRouter()


# 新增登录请求的模型
class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/token")
async def login_for_access_token(
        login_data: LoginRequest,  # 改用 JSON 请求体
        db: Session = Depends(get_db)
):
    user = authenticate_user(db, login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "nick_name": current_user.nick_name
    }
