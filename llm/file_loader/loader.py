from abc import ABC, abstractmethod
from io import StringIO
from typing import Optional

import yaml
from langchain_core.documents import Document
from pydantic import FilePath

from llm.schemas import ArticleBlock, MarkdownMeta


class BaseFileLoader(ABC):
    """文件夹加载器的基类

    对于所有类型的文件，以此类为基础，
    实现加载和保存为markdown格式这两个基础功能

    Attributes:
        article:
        file_meta:
    """
    article: list[ArticleBlock]
    file_meta: Optional[MarkdownMeta] = None
    keep_title: bool = True

    @abstractmethod
    def load(self, origin_file_path: FilePath) -> tuple[Optional[MarkdownMeta], list[Document]]:
        """

        Args:
            origin_file_path: 待加载的源文件路径

        Returns:

        """
        raise NotImplementedError

    def save_md(self, md_path: FilePath) -> None:
        """将文档存储为markdown格式

        若存在meta信息，则会以Front matter的形式进行存储

        Args:
            md_path: 输出的markdown文件路径
        """
        with open(md_path, 'w', encoding='utf-8') as f:
            if self.file_meta:
                meta_stream = StringIO()
                meta_str = self.__meta_to_stream(meta_stream)
                meta_stream.close()
                f.write(meta_str)

            for block in self.article:
                if 1 <= block.text_level <= 6:
                    # 标题行
                    f.write(f'{"#" * block.text_level} {block.text}  \n')  # pylint: disable=inconsistent-quotes
                else:
                    # 正文行
                    f.write(f'{block.text}  \n')

    def __meta_to_stream(self, stream: StringIO) -> str:
        stream.write('---\t\n')
        yaml.dump(self.file_meta.model_dump(), stream, sort_keys=False, width=900)
        stream.write('---\t\n')

        return stream.getvalue()
