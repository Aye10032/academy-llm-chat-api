import argparse
import sys
from argparse import Namespace

from loguru import logger
from sqlmodel import select

from app.core.config import get_settings
from app.core.security import get_password_hash
from app.db.session import get_simple_session, create_db_and_tables
from app.models import *  # pylint: disable=wildcard-import
from app.schemas.user import UserRole, UserPublic

logger.remove()
handler_id = logger.add(sys.stderr, level='DEBUG')
logger.add('log/init_database.log')


def init_user(email: str, password: str):
    db = get_simple_session()
    create_db_and_tables()

    statement = select(UserTable).where(UserTable.email == email)
    check_user = db.exec(statement).first()
    if check_user:
        logger.warning('User already exist')
        logger.warning(UserPublic.model_validate(check_user))
        return

    test_user = UserTable(
        email=email,
        username='Admin',
        hashed_password=get_password_hash(password),
        is_active=True,
        role=UserRole.ADMIN
    )

    db.add(test_user)
    db.commit()
    logger.info('Done')

    db.close()


def load_args(args: Namespace):
    setting = get_settings()

    if args.user:
        logger.info('Create admin profile...')
        init_user(setting.server.INIT_USER, setting.server.INIT_PASSWORD)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='database init')
    parser.add_argument(
        '--user',
        '-U',
        action='store_true',
        help='Init default admin profile'
    )
    parser.add_argument(
        '--knowledge_base',
        '-K',
        nargs='?',
        const=-1,
        type=int,
        help='Initialize a specific collection, starting from 0.'
    )
    load_args(parser.parse_args())
