from fastapi import APIRouter, Depends
from app.api.authentication import get_current_user
from app.models.user import User

router = APIRouter()


# 公开路由，不需要认证
@router.get("/hello")
async def hello_world():
    return {"message": "Hello World!"}


# 需要认证的路由
@router.get("/hello/protected")
async def hello_protected(current_user: User = Depends(get_current_user)):
    return {
        "message": f"Hello {current_user.nick_name}!",
        "email": current_user.email
    }
