from app.models.user import Base
from app.db.session import engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
