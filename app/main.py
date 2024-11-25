from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.endpoints import auth
from app.db.session import create_db_and_tables


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 启动时执行
    create_db_and_tables()
    yield

    # 关闭时执行
    pass


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
