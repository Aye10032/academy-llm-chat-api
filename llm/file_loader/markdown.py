import re
from datetime import datetime
from typing import Optional, Any

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import FilePath
from pydantic_core import ValidationError

from llm.file_loader.loader import BaseFileLoader
from llm.schemas.markdown import MarkdownMeta, MARKDOWN


class MarkdownLoader(BaseFileLoader):

    def load(self):
        pass

    def save_md(self, md_path: FilePath):
        pass


def extract_title_list(md_docs: list[Document]) -> str:
    """将markdown文本的标题提取为列表文本

    Args:
        md_docs: 分割结束的Markdown Document列表

    Returns:
        markdown列表格式的标题提取
    """

    titles = []
    for doc in md_docs:
        lines = doc.page_content.split('\n')
        while lines and re.match(r'^(#+)\s*(.+)$', lines[0]):
            titles.append(lines.pop(0))

    title_list = []
    for heading in titles:
        level = heading.count('#')
        text = heading.replace('#', '').strip()
        indent = '    ' * (level - 1)
        title_list.append(f"{indent}- {text}")

    return '\n'.join(title_list)


def split_md_text(
        md_text: str,
        *,
        additional_metadata: Optional[dict[str, Any]] = None
) -> list[Document]:
    """对markdown格式的文本进行分块

    对于markdown文件，采用yaml头部存储meta信息。
    同时，会将markdown的标题进行提取并作为一个列表添加进文档列表中。

    Args:
        md_text: 原始的markdown文本
        additional_metadata: 额外添加到文档的meta信息

    Returns:
        分块后的Langchain Document列表
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ('#', 'title'),
            ('##', 'section'),
            ('###', 'subtitle'),
            ('####', 'subtitle'),
            ('#####', 'subtitle'),
            ('######', 'subtitle'),
        ],
        strip_headers=False
    )
    r_splitter = RecursiveCharacterTextSplitter(
        separators=['\n'],
        keep_separator=False
    )

    head_split_docs = md_splitter.split_text(md_text)

    if head_split_docs[0].page_content.startswith('---'):
        yaml_text = head_split_docs.pop(0).page_content.replace('---', '')
        data = yaml.load(yaml_text, Loader=yaml.FullLoader)
        try:
            markdown_meta = MarkdownMeta(**data)
        except ValidationError:
            markdown_meta = None
    else:
        markdown_meta = None

    if not markdown_meta:
        markdown_meta = MarkdownMeta(
            title=head_split_docs[0].metadata['title'],
            year=datetime.now().year,
            source_type=MARKDOWN
        )

    for doc in head_split_docs:
        if 'abstract' in doc.metadata['section'].lower():
            doc.metadata.update({
                'author': markdown_meta.author,
                'year': markdown_meta.year,
                'type': 'abstract',
                'source': markdown_meta.source,
                'source_type': markdown_meta.source_type
            })
        else:
            doc.metadata.update({
                'author': markdown_meta.author,
                'year': markdown_meta.year,
                'type': 'content',
                'source': markdown_meta.source,
                'source_type': markdown_meta.source_type
            })

        if additional_metadata:
            doc.metadata.update(additional_metadata)

    md_docs = r_splitter.split_documents(head_split_docs)

    title_list = extract_title_list(md_docs)
    title_list_doc = Document(
        page_content=title_list,
        metadata=md_docs[-1].metadata
    )
    title_list_doc.metadata.update({
        'section': '',
        'type': 'toc'
    })
    md_docs.append(title_list_doc)

    return md_docs


def load_markdown(md_file: FilePath) -> list[Document]:
    with open(md_file, 'r', encoding='utf-8') as f:
        md_text = f.read()

    return split_md_text(md_text)


def load_markdown_external(md_file: FilePath):
    with open(md_file, 'r', encoding='utf-8') as f:
        md_text = f.read()

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ('#', 'title'),
            ('##', 'section'),
            ('###', 'subtitle'),
            ('####', 'subtitle'),
            ('#####', 'subtitle'),
            ('######', 'subtitle'),
        ],
        strip_headers=False
    )


def main() -> None:
    docs = load_markdown('../../test/rag_mds/5.4 Agent中的提示词优化.md')
    print(docs)
    # for doc in docs:
    #     print('========================')
    #     print(doc)


if __name__ == '__main__':
    main()
