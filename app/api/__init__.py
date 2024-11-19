from fastapi import APIRouter
from app.api.endpoints import auth, example

# 创建一个主路由器
api_router = APIRouter()

# 注册所有子路由器
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(example.router, prefix="/example", tags=["example"])
