from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger

from app.core.config import get_settings
from app.api.v1.endpoints import auth, rag, user, write
from app.db.session import create_db_and_tables
from app.utils.logger import init_logging


@asynccontextmanager
async def lifespan(main_app: FastAPI):  # pylint: disable=unused-argument
    create_db_and_tables()
    yield

    logger.info('server stop')


app = FastAPI(
    title=get_settings().PROJECT_NAME,
    version=get_settings().VERSION,
    lifespan=lifespan
)

app.include_router(auth.router, prefix='/api/v1/auth', tags=['auth'])
app.include_router(user.router, prefix='/api/v1/user', tags=['user'])
app.include_router(rag.router, prefix='/api/v1/rag', tags=['rag'])
app.include_router(write.router, prefix='/api/v1/write', tags=['write'])


def main() -> None:
    config = uvicorn.Config(
        'main:app',
        host=get_settings().server.network.SERVICE_HOST_IP,
        port=get_settings().server.network.SERVICE_HOST_PORT,
        access_log=True,
        workers=1,
    )
    server = uvicorn.Server(config)
    init_logging()

    try:
        server.run()
    except KeyboardInterrupt as e:
        logger.error(f'server closed for {e}')


if __name__ == '__main__':
    main()
