import re
from datetime import datetime
from typing import Optional, Any

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import FilePath

from llm.schemas.markdown import MarkdownMeta, MARKDOWN


def extract_title_list(md_text: str) -> str:
    """将markdown文本的标题提取为列表文本

    Args:
        md_text: markdown文本

    Returns:
        markdown列表格式的标题提取
    """
    titles = re.findall(r'^(#+)\s*(.+)$', md_text, re.MULTILINE)
    title_list = '\n'.join([
        f'{"    " * (len(title[0]) - 1)}- {title[1]}'  # pylint: disable=inconsistent-quotes
        for title in titles
    ])

    print(title_list)

    return title_list


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
            ('####', 'subtitle')
        ]
    )
    r_splitter = RecursiveCharacterTextSplitter(
        separators=['\n'],
        keep_separator=False
    )

    head_split_docs = md_splitter.split_text(md_text)

    if head_split_docs[0].page_content.startswith('---'):
        yaml_text = head_split_docs.pop(0).page_content.replace('---', '')
        data = yaml.load(yaml_text, Loader=yaml.FullLoader)
        markdown_meta = MarkdownMeta(**data)
    else:
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

    title_list = extract_title_list(md_text)
    title_list_doc = Document(
        page_content=title_list,
        metadata=md_docs[-1].metadata
    )
    # TODO: type name
    title_list_doc.metadata.update({'type': ''})
    md_docs.append(title_list_doc)

    return md_docs


def load_markdown(md_file: FilePath) -> list[Document]:
    with open(md_file, 'r', encoding='utf-8') as f:
        md_text = f.read()

    return split_md_text(md_text)


def main() -> None:
    docs = load_markdown('../../test/10.1002@advs.202207497.md')
    print(docs)


if __name__ == '__main__':
    main()
