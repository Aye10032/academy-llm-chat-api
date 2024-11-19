from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.db.init_db import init_db
from app.api import api_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 启动时执行
    init_db()
    yield

    # 关闭时执行
    pass


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# 注册所有路由
app.include_router(api_router, prefix="/api")
