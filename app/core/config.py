import os
import secrets
import shutil
from functools import lru_cache
from typing import Type, Self, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict, model_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, TomlConfigSettingsSource


class ServerNetworkSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    SERVICE_HOST_IP: str
    SERVICE_HOST_PORT: int
    PROXY: str


class ServerSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    DATABASE_URL: str
    LOGGING_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR']
    INIT_USER: str
    INIT_PASSWORD: str

    network: ServerNetworkSetting


class BaseModelSetting(BaseModel):
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


class KnowledgeBaseSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    VECTOR_DB_FILE: str
    DOC_DB_FILE: str


class RetrieverSetting(BaseModel):
    embedding: BaseModelSetting
    reranker: BaseModelSetting
    knowledge_base: KnowledgeBaseSetting


class BaseLLMSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    USE_PROXY: bool
    BASE_URL: Optional[str] = None
    API_KEY: SecretStr
    SECRET_KEY: Optional[SecretStr] = None


class LLMSetting(BaseModel):
    openai: BaseLLMSetting
    zhipu: BaseLLMSetting


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
    if not os.path.exists('config.toml'):
        shutil.copy('config.example.toml.', 'config.toml')
        logger.error('建配置文件"config.toml"不存在，已自动初始化，请完善相关设置')
        exit()

    settings = Settings()
    return settings


def main() -> None:
    setting = Settings()
    print(setting)


if __name__ == '__main__':
    main()
