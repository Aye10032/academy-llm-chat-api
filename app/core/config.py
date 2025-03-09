import os
import secrets
import shutil
from functools import lru_cache
from typing import Any, Literal, Optional, Self, Type

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class ServerNetworkSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    SERVICE_HOST_IP: str
    SERVICE_HOST_PORT: int
    USE_PROXY: bool
    PROXY: str


class ServerSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    DATABASE_URL: str
    GRAPH_STORE_URL: str
    LOGGING_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR']
    INIT_USER: str
    INIT_PASSWORD: str
    TEMP_DIR: str

    network: ServerNetworkSetting

    @model_validator(mode='after')
    def create_temp_dir(self) -> Self:
        os.makedirs(self.TEMP_DIR, exist_ok=True)

        return self


class GrobidSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    GROBID_SERVER: str
    SERVICE: Literal['processFulltextDocument']
    BATCH_SIZE: int
    SLEEP_TIME: int
    TIMEOUT: int
    COORDINATES: list[str] = Field(default_factory=list)
    MULTI_PROCESS: int


class FileLoaderSetting(BaseModel):
    grobid: GrobidSetting


class BaseModelSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    MODEL: str
    SAVE_LOCAL: bool
    FP16: bool
    NORMALIZE: bool = False
    DEVICE: str = 'cpu'
    LOCAL_PATH: str = Field(default='', init=False)

    @model_validator(mode='after')
    def set_local_path(self) -> Self:
        local_path = os.path.join('data', 'model', self.MODEL)
        self.LOCAL_PATH = local_path  # pylint: disable=invalid-name

        return self


class MilvusSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    URI: str
    USERNAME: str
    PASSWORD: str
    TOKEN: str
    SECURE: bool

    def get_conn_args(self, db_name: str = 'default') -> dict[str, Any]:
        return {
            'uri': self.URI,
            'user': self.USERNAME,
            'password': self.PASSWORD,
            'token': self.TOKEN,
            'db_name': db_name,
            'secure': self.SECURE,
        }


class KnowledgeBaseSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    STORE_PATH: str
    DOC_URL: str
    milvus: MilvusSetting

    @model_validator(mode='after')
    def create_path(self) -> Self:
        os.makedirs(self.STORE_PATH, exist_ok=True)

        return self


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
    deepseek: BaseLLMSetting


class SearchSetting(BaseModel):
    model_config = ConfigDict(alias_generator=lambda field_name: field_name.lower())

    SERPER_API: str = ''
    JINA_API: str = ''


class JinaSetting(BaseModelSetting):
    JINA_API: str
    USE_LOCAL_MODEL: bool


class ToolSetting(BaseModel):
    search: SearchSetting
    jina: JinaSetting


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file='config.toml', alias_generator=lambda field_name: field_name.lower()
    )

    PROJECT_NAME: str = 'Academic LLM Chat API'
    VERSION: str = '1.0.0'

    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    server: ServerSetting
    fileloader: FileLoaderSetting
    retriever: RetrieverSetting
    llm: LLMSetting
    tool: ToolSetting

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
