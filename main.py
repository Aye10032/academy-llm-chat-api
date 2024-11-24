from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.db.init_db import init_db
from app.api.endpoints import auth, example, users


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

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(example.router, prefix="/api/example", tags=["example"])
app.include_router(users.router, prefix="/api/user", tags=["users"])
