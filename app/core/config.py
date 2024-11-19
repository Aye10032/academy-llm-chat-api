from pydantic_settings import BaseSettings
import secrets


class Settings(BaseSettings):
    """
    Settings class to hold configuration variables for the application.

    Attributes:
        PROJECT_NAME (str): The name of the project.
        VERSION (str): The version of the project.
        API_V1_STR (str): The base URL for version 1 of the API.
        SECRET_KEY (str): A secret key for cryptographic operations.
        ACCESS_TOKEN_EXPIRE_MINUTES (int): The expiration time for access tokens in minutes.
        SQLITE_DATABASE_URL (str): The database URL for SQLite.
    """

    PROJECT_NAME: str = "Academy LLM Chat API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    SQLITE_DATABASE_URL: str = "sqlite:///./sql_app.db"

    class Config:
        """
        Configuration class for additional settings.

        Attributes:
            case_sensitive (bool): Whether the settings are case-sensitive.
        """
        case_sensitive = True


settings = Settings()
