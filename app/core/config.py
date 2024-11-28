import secrets
from functools import lru_cache
from typing import Type

from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, TomlConfigSettingsSource


class ServerSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    SERVICE_HOST_IP: str
    SERVICE_HOST_PORT: int

    SQLITE_DATABASE_URL: str

    LOGGING_LEVEL: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file='config.toml',
        case_sensitive=False
    )

    PROJECT_NAME: str = "Academic LLM Chat API"
    VERSION: str = "1.0.0"

    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    server_setting: ServerSetting

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)


@lru_cache
def get_settings():
    settings = Settings()
    return settings


def main() -> None:
    setting = Settings()
    print(setting)


if __name__ == '__main__':
    main()
