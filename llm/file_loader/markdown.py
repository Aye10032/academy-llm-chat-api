import re
from datetime import datetime
from uuid import uuid4

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from loguru import logger
from pydantic import FilePath
from pydantic_core import ValidationError

from llm.file_loader.loader import BaseFileLoader
from llm.schemas import ArticleBlock
from llm.schemas.markdown import MarkdownMeta, SourceType, FileSource


class MarkdownLoader(BaseFileLoader):
    """对于markdown格式文件的加载器

    Examples:
        ```python
        md_loader = MarkdownLoader(keep_title=True, add_toc=True)
        meta, docs = md_loader.load('test.md')
        ```
    """

    def load(
        self, origin_file_path: FilePath, **kwargs
    ) -> tuple[MarkdownMeta, list[Document]]:
        """从markdown文件加载文档

        Args:
            origin_file_path: 原始markdown文件

            **kwargs:
                additional_metadata: 若有额外需要添加进文档的meta字段，通过此参数以字典形式传入

        Returns:
            一个元组，第一个元素为文档的meta信息，第二个为langchain格式的Document对象列表
        """
        assert origin_file_path.lower().endswith('.md')

        with open(origin_file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()

        # 加载meta信息
        if md_text.startswith('---'):
            yaml_text = md_text.lstrip('---').split('---')[0].strip()
            data = yaml.load(yaml_text, Loader=yaml.FullLoader)
            try:
                self.file_meta = MarkdownMeta(**data)
            except ValidationError:
                pass
            finally:
                md_text = md_text.lstrip('---').split('---', 1)[1].strip()

        # 章节分割
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', 'title'),
                ('##', 'section'),
                ('###', 'section_3'),
                ('####', 'section_4'),
                ('#####', 'section_5'),
                ('######', 'section_6'),
            ],
            strip_headers=False,
        )

        head_split_docs = md_splitter.split_text(md_text)

        if not self.file_meta:
            self.file_meta = MarkdownMeta(
                title=head_split_docs[0].metadata['title'],
                year=datetime.now().year,
                source=[FileSource(source_url='', source_type=SourceType.MARKDOWN)],
            )

        # 存储为统一文档块，并提取标题
        title_list = []
        file_uid = str(uuid4())
        has_abstract = False
        for doc in head_split_docs:
            lines = doc.page_content.split('\n')
            while lines and re.match(r'^(#+)\s*(.+)$', lines[0]):
                heading = lines.pop(0)
                level = heading.count('#')
                text = heading.replace('#', '').strip()

                self.article.append(ArticleBlock(text=text, text_level=level))
                title_list.append(f'{"    " * (level - 1)}- {text}')

            self.article.append(ArticleBlock(text='\n'.join(lines)))

            if not self.keep_title:
                doc.page_content = '\n'.join(lines)

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

            if 'additional_metadata' in kwargs:
                doc.metadata.update(kwargs.get('additional_metadata'))

        # 添加目录文档块
        if self.add_toc:
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
            if 'additional_metadata' in kwargs:
                title_list_doc.metadata.update(kwargs.get('additional_metadata'))
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
            head_split_docs.append(abstract_doc)

        return self.file_meta, head_split_docs
