import os
import secrets
from functools import lru_cache
from typing import Type, Self, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, TomlConfigSettingsSource


class ServerSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    SERVICE_HOST_IP: str
    SERVICE_HOST_PORT: int

    SQLITE_DATABASE_URL: str

    LOGGING_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR']


class ModelBaseSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    MODEL: str
    SAVE_LOCAL: bool
    FP16: bool
    NORMALIZE: bool
    DEVICE: str
    LOCAL_PATH: str = Field(default="", init=False)

    @model_validator(mode='after')
    def set_local_path(self) -> Self:
        local_path = os.path.join('data', 'model', self.MODEL)
        self.LOCAL_PATH = local_path

        return self


class RetrieverSetting(BaseModel):
    embedding: ModelBaseSetting
    reranker: ModelBaseSetting


class LLMBaseSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    USE_PROXY: bool
    BASE_URL: Optional[str] = None
    API_KEY: SecretStr
    SECRET_KEY: Optional[SecretStr] = None


class LLMSetting(BaseModel):
    openai: LLMBaseSetting
    zhipu: LLMBaseSetting


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file='config.toml',
        alias_generator=lambda field_name: field_name.lower()
    )

    PROJECT_NAME: str = "Academic LLM Chat API"
    VERSION: str = "1.0.0"

    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    PROXY: str

    server: ServerSetting
    retriever: RetrieverSetting
    llm: LLMSetting

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
def get_settings() -> Settings:
    settings = Settings()
    return settings


def main() -> None:
    setting = Settings()
    print(setting)


if __name__ == '__main__':
    main()
