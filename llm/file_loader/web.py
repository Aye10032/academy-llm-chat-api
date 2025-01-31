import datetime
import random
import re
from typing import Optional, Self, Any

import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from loguru import logger
from pydantic import AnyHttpUrl, model_validator, BaseModel, Field, FilePath
from requests import HTTPError

from app.core.config import get_settings
from app.utils.network import retry, download_html
from llm.core.model import load_jina_reader
from llm.file_loader.loader import BaseFileLoader
from llm.schemas import MarkdownMeta, ArticleBlock
from llm.schemas.markdown import FileSource, SourceType

network_setting = get_settings().server.network


class SimpleWebLoader(BaseFileLoader):

    def load(self, origin_file_path: AnyHttpUrl, **kwargs) -> tuple[MarkdownMeta, list[Document]]:
        if network_setting.USE_PROXY:
            loader = WebBaseLoader(
                origin_file_path,
                header_template={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                },
                encoding='utf-8',
                proxies={
                    'http': network_setting.PROXY,
                    'https': network_setting.PROXY
                }
            )
        else:
            loader = WebBaseLoader(
                origin_file_path,
                header_template={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                },
                encoding='utf-8'
            )

        docs = loader.load()
        markdownify = MarkdownifyTransformer()
        docs_transform = markdownify.transform_documents(docs)
        assert len(docs_transform) == 1

        now_time = datetime.datetime.now()
        title = docs_transform[0].metadata['title']
        self.file_meta = MarkdownMeta(
            title=kwargs.get('title', title),
            author='',
            year=kwargs.get('year', now_time.year),
            source=[
                FileSource(source_url=origin_file_path, source_type=SourceType.WEB)
            ]
        )

        self.article = [ArticleBlock(text=self.file_meta.title, text_level=1)]
        self.article.extend([
            ArticleBlock(text=doc.page_content)
            for doc in docs_transform
        ])

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', 'title'),
                ('##', 'section'),
                ('###', 'section_3'),
                ('####', 'section_4'),
                ('#####', 'section_5'),
                ('######', 'section_6'),
            ],
            strip_headers=False
        )

        head_split_docs = md_splitter.split_text(docs_transform[0].page_content)
        for doc in head_split_docs:
            if not 'section' in doc.metadata:
                doc.metadata['section'] = 'content'

            doc.metadata.update({
                'title': self.file_meta.title,
                'author': self.file_meta.author,
                'year': self.file_meta.year,
                'type': 'content',
                'source': self.file_meta.model_dump()['source']
            })

            if 'additional_metadata' in kwargs:
                doc.metadata.update(kwargs.get('additional_metadata'))

        return self.file_meta, head_split_docs


class JinaData(BaseModel):
    title: str
    description: Optional[str] = None
    date: Optional[str] = None
    url: AnyHttpUrl
    content: str
    usage: dict[str, Any] = Field(default_factory=dict)


class JinaWebLoader(BaseFileLoader):
    """网页加载器

    使用jina reader进行网页的解析。若不设置API，则请求会被限速，
    具体可见：https://jina.ai/reader/

    Attributes:
        jina_api: jina api密钥，无需显式设置，会自动从配置文件读取
        proxy: 对解析网页使用的代理，若留空则不使用代理。
            无需显式设置，会自动从配置文件读取
    """
    jina_api: Optional[str] = None
    proxy: Optional[AnyHttpUrl] = None

    use_local_model: bool = Field(default=True, init=False)

    @model_validator(mode='after')
    def check_jina(self) -> Self:
        self.use_local_model = get_settings().tool.jina.USE_LOCAL_MODEL

        if self.jina_api:
            return self

        if jina_api := get_settings().tool.jina.JINA_API:
            self.jina_api = jina_api
        else:
            logger.warning('未设置jina api，解析请求将被限速')

        return self

    @model_validator(mode='after')
    def check_proxy(self):
        if self.proxy:
            return self

        self.proxy = network_setting.PROXY if network_setting.USE_PROXY else None
        return self

    @retry(delay=random.randint(3, 5))
    def _read_webpage(self, url: AnyHttpUrl) -> JinaData:
        if self.use_local_model:
            html_text = download_html(url)
            html_reader = load_jina_reader()

            json_data = html_reader.html_to_json(html_text)
        else:
            api_url = f'https://r.jina.ai/{url}'
            headers = {
                'Accept': 'application/json',
                'X-Remove-Selector': 'header, .class, #id',
                'X-Retain-Images': 'none',
                'X-Timeout': '120'
            }
            if self.jina_api:
                headers['Authorization'] = f'Bearer {self.jina_api}'
            # if self.proxy:
            #     headers['X-Proxy-Url'] = self.proxy

            response = requests.get(api_url, headers=headers, timeout=120)

            if response.status_code != 200:
                raise HTTPError(f'请求失败 code:{response.status_code}')

            json_data = response.json()['data']

        data = JinaData.model_validate(json_data)
        return data

    def load(self, origin_file_path: AnyHttpUrl, **kwargs) -> tuple[MarkdownMeta, list[Document]]:
        doc_data = self._read_webpage(origin_file_path)

        self.file_meta = MarkdownMeta(
            title=doc_data.title,
            author='',
            year=kwargs.get('year', -1),
            source=[
                FileSource(source_url=origin_file_path, source_type=SourceType.WEB)
            ]
        )

        self.article = [
            ArticleBlock(text=doc_data.title, text_level=1),
            ArticleBlock(text=doc_data.content)
        ]

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', 'title'),
                ('##', 'section'),
                ('###', 'section_3'),
                ('####', 'section_4'),
                ('#####', 'section_5'),
                ('######', 'section_6'),
            ],
            strip_headers=False
        )

        head_split_docs = md_splitter.split_text(doc_data.content.strip().lstrip('```markdown').rstrip('```'))
        for doc in head_split_docs:
            if not 'section' in doc.metadata:
                doc.metadata['section'] = 'content'

            doc.metadata.update({
                'title': self.file_meta.title,
                'author': self.file_meta.author,
                'year': self.file_meta.year,
                'type': 'content',
                'source': self.file_meta.model_dump()['source']
            })

            if 'additional_metadata' in kwargs:
                doc.metadata.update(kwargs.get('additional_metadata'))

        if doc_data.description:
            abstract_doc = Document(
                page_content=doc_data.description,
                metadata={
                    'title': self.file_meta.title,
                    'section': self.abstract_key,
                    'author': self.file_meta.author,
                    'year': self.file_meta.year,
                    'type': 'abstract',
                    'source': self.file_meta.model_dump()['source']
                }
            )
            if 'additional_metadata' in kwargs:
                abstract_doc.metadata.update(kwargs.get('additional_metadata'))
            head_split_docs.append(abstract_doc)

        return self.file_meta, head_split_docs
