from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger

from app.core.config import get_settings
from app.api.v1.endpoints import auth
from app.db.session import create_db_and_tables
from app.utils.logger import init_logging

init_logging()


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


def main() -> None:
    config = uvicorn.Config(
        "main:app",
        host=get_settings().SERVEICE_HOST_IP,
        port=get_settings().SERVEICE_HOST_PORT,
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
