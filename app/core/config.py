import secrets
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Settings class to hold configuration variables for the application.

    Attributes:
        PROJECT_NAME (str): The name of the project.
        VERSION (str): The version of the project.
        SECRET_KEY (str): A secret key for cryptographic operations.
        ACCESS_TOKEN_EXPIRE_MINUTES (int): The expiration time for access tokens in minutes.
        SQLITE_DATABASE_URL (str): The database URL for SQLite.
    """

    PROJECT_NAME: str = "Academic LLM Chat API"
    VERSION: str = "1.0.0"

    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    SERVEICE_HOST_IP: str = "127.0.0.1"
    SERVEICE_HOST_PORT: int = 8000

    SQLITE_DATABASE_URL: str = "sqlite:///./sql_app.db"

    LOGGING_LEVEL: str = "INFO"

    class Config:
        """
        Configuration class for additional settings.

        Attributes:
            case_sensitive (bool): Whether the settings are case-sensitive.
        """
        case_sensitive = True


@lru_cache
def get_settings():
    settings = Settings()
    return settings
