import io
from abc import ABC, abstractmethod
from io import StringIO
from typing import Any, Optional
from uuid import uuid4

import yaml
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import AnyHttpUrl, BaseModel, Field, FilePath

from llm.schemas import ArticleBlock, MarkdownMeta
from llm.schemas.markdown import FileSource

SYS_PROMPT = """你是一个专业的文本摘要生成器。你的任务是根据用户提供的文章，生成简洁、准确、信息量丰富的摘要。摘要应：
1.  抓住文章的核心思想和关键信息。
2.  避免复制原文中的长句，尽量使用自己的语言进行概括。
3.  长度应控制在原文的 20%-30% 左右。
4.  如果原文包含多个主题，应在摘要中有所体现。
5.  使用清晰简洁的语言，避免使用过于专业的术语。"""

PROMPT = """请为以下文章生成一段摘要：
{article}"""


class FileLoadError(Exception):
    pass


class BaseFileLoader(BaseModel, ABC):
    """文件夹加载器的基类

    对于所有类型的文件，以此类为基础，
    实现加载和保存为markdown格式这两个基础功能

    Attributes:
        article: 统一格式文档块列表
        file_meta: 文档的meta信息

        keep_title: 是否在分块文档中保留标题
        add_toc: 是否返回全文目录信息

        abstract_key: 摘要章节检测关键字
        generate_abstract: 当文档不存在摘要章节时，通过大模型生成摘要
        llm: 若generate_abstract为true，则必须传入一个可用的大模型，用于生成总结
    """

    article: list[ArticleBlock] = Field(default_factory=list)
    file_meta: Optional[MarkdownMeta] = None

    keep_title: bool = True
    add_toc: bool = True

    # 默认会使用文档的section（也就是二级标题）进行检测，若改标题含有摘要关键字，则会将这一段标记为摘要。
    # 默认的摘要关键字是abstract，可以进行修改。
    # 若不存在关键字，则不会标记摘要段落。这会使得此文本无法被摘要搜索索引。
    # 为了解决这个问题，可以将generate_abstract设为真，此时会使用大模型对全文进行总结，生成一段摘要文本。
    # 例：
    # llm = load_glm4_flash()
    # md_loader = MarkdownLoader(generate_abstract=True, llm=llm)
    # meta, docs = md_loader.load('test.md')
    abstract_key: str = 'abstract'
    abstract_level: str = 'section'
    generate_abstract: bool = False
    llm: Optional[BaseChatModel] = None
    sys_prompt: str = SYS_PROMPT
    prompt: str = PROMPT

    @abstractmethod
    def load(
        self, origin_file_path: FilePath | AnyHttpUrl, **kwargs
    ) -> tuple[MarkdownMeta, list[Document]]:
        raise NotImplementedError

    def save_md(self, md_path: FilePath) -> None:
        """将文档存储为markdown格式

        若存在meta信息，则会以Front matter的形式进行存储

        Args:
            md_path: 输出的markdown文件路径
        """
        with open(md_path, 'w', encoding='utf-8') as f:
            if self.file_meta:
                meta_str = self.__meta_to_str()
                f.write(meta_str)

            f.write(self.__article_to_str())

        self.file_meta = None
        self.article = []

    def update_source(self, new_source: list[FileSource]):
        if self.file_meta:
            self.file_meta.source = new_source

    def __meta_to_str(self) -> str:
        stream = StringIO()
        stream.write('---\t\n')
        yaml.dump(self.file_meta.model_dump(), stream, sort_keys=False, width=900)
        stream.write('---\t\n')

        text = stream.getvalue()
        stream.close()
        return text

    def __article_to_str(self) -> str:
        return '  \n'.join(
            [
                block.text if block.text_level == 0 else f'{"#" * block.text_level} {block.text}'  # pylint: disable=inconsistent-quotes
                for block in self.article
            ]
        )

    def _conclude_article(self) -> str:
        from llm.core.model import load_llm

        if self.llm is None:
            self.llm = load_llm('glm-4-flash')

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.sys_prompt),
                ('human', self.prompt),
            ]
        )
        chain = prompt | self.llm

        result = chain.invoke({'article': self.__article_to_str()})

        return result.content

    def _article_to_doc(
        self, *, additional_metadata: Optional[dict[str, Any]] = None
    ) -> list[Document]:
        md_stream = io.StringIO()
        for section in self.article:
            if section.text_level == 0:
                md_stream.write(f'{section.text}\t\n')
            else:
                md_stream.write('#' * section.text_level)
                md_stream.write(f' {section.text}\t\n')
        md_text = md_stream.getvalue()
        md_stream.close()

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', 'title'),
                ('##', 'section'),
                ('###', 'section_3'),
                ('####', 'section_4'),
                ('#####', 'section_5'),
                ('######', 'section_6'),
            ],
            strip_headers=not self.keep_title,
        )
        head_split_docs = md_splitter.split_text(md_text)

        file_uid = str(uuid4())
        has_abstract = False
        for doc in head_split_docs:
            if (
                self.abstract_level in doc.metadata
                and self.abstract_key in doc.metadata[self.abstract_level].lower()
            ):
                doc.metadata.update(
                    {
                        'author': self.file_meta.author,
                        'year': self.file_meta.year,
                        'type': 'abstract',
                        'source': self.file_meta.model_dump()['source'],
                        'file_id': file_uid,
                    }
                )
                has_abstract = True
            else:
                doc.metadata.update(
                    {
                        'author': self.file_meta.author,
                        'year': self.file_meta.year,
                        'type': 'content',
                        'source': self.file_meta.model_dump()['source'],
                        'file_id': file_uid,
                    }
                )

            if additional_metadata:
                doc.metadata.update(additional_metadata)

        # 添加目录文档块
        if self.add_toc:
            title_list = []
            for section in self.article:
                level = section.text_level
                text = section.text
                if level == 0:
                    continue

                title_list.append(f'{"    " * (level - 1)}- {text}')  # pylint: disable=inconsistent-quotes
            title_list_doc = Document(
                page_content='\n'.join(title_list),
                metadata={
                    'title': self.file_meta.title,
                    'section': 'toc',
                    'author': self.file_meta.author,
                    'year': self.file_meta.year,
                    'type': 'toc',
                    'source': self.file_meta.model_dump()['source'],
                    'file_id': file_uid,
                },
            )

            if additional_metadata:
                title_list_doc.metadata.update(additional_metadata)

            head_split_docs.append(title_list_doc)

        if not has_abstract and self.generate_abstract:
            abstract = self._conclude_article()
            abstract_doc = Document(
                page_content=abstract,
                metadata={
                    'title': self.file_meta.title,
                    'section': self.abstract_key,
                    'author': self.file_meta.author,
                    'year': self.file_meta.year,
                    'type': 'abstract',
                    'source': self.file_meta.model_dump()['source'],
                    'file_id': file_uid,
                },
            )

            if additional_metadata:
                abstract_doc.metadata.update(additional_metadata)

            head_split_docs.append(abstract_doc)

        return head_split_docs
