import os.path
import shutil
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger

if not os.path.exists('config.toml'):
    shutil.copy('config.example.toml.', 'config.toml')
    logger.error('建配置文件"config.toml"不存在，已自动初始化，请完善相关设置')
    exit()

from app.core.config import get_settings
from app.api.v1.endpoints import auth, chat
from app.db.session import create_db_and_tables
from app.utils.logger import init_logging


@asynccontextmanager
async def lifespan(_app: FastAPI):
    create_db_and_tables()
    yield

    pass


app = FastAPI(
    title=get_settings().PROJECT_NAME,
    version=get_settings().VERSION,
    lifespan=lifespan
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])


def main() -> None:
    config = uvicorn.Config(
        "main:app",
        host=get_settings().server.SERVICE_HOST_IP,
        port=get_settings().server.SERVICE_HOST_PORT,
        access_log=True,
        workers=1,
        reload=True
    )
    server = uvicorn.Server(config)
    init_logging()

    try:
        server.run()
    except KeyboardInterrupt as e:
        logger.error("server closed")


if __name__ == '__main__':
    main()
