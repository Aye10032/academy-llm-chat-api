import random
import re
from enum import StrEnum
from typing import Literal, Any

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from loguru import logger
from pydantic import FilePath
from requests import RequestException
from urllib3.exceptions import ResponseError

from app.core.config import GrobidSetting
from app.utils.network import retry
from llm.file_loader.loader import BaseFileLoader
from llm.schemas import MarkdownMeta, ArticleBlock
from llm.schemas.markdown import FileSource, SourceType


class ConsolidateHeader(StrEnum):
    NO_CONSOLIDATION = '0'
    ALL_METADATA = '1'
    CITATION_AND_DOI = '2'
    DOI_ONLY = '3'


class ConsolidateCitations(StrEnum):
    NO_CONSOLIDATION = '0'
    ALL_METADATA = '1'
    CITATION_AND_DOI = '2'


class ConsolidateFunders(StrEnum):
    NO_CONSOLIDATION = '0'
    ALL_METADATA = '1'
    CITATION_AND_DOI = '2'


class GrobidConnector:
    """Grobid pdf解析服务封装"""

    def __init__(self, config: GrobidSetting):
        self.server_url = f'{config.GROBID_SERVER}/api/{config.SERVICE}'
        self.check_url = f'{config.GROBID_SERVER}/api/isalive'
        self.coordinates = config.COORDINATES
        self.timeout = config.TIMEOUT
        self.batch_size = config.BATCH_SIZE
        self.max_works = config.MULTI_PROCESS

    def __enter__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/xml'
        })

        self._check_server_status()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def _check_server_status(self):
        try:
            response = requests.get(self.check_url, timeout=self.timeout)
            response.raise_for_status()
        except RequestException as e:
            logger.error(f'[{e}]: Grobid server is unavailable.')
            raise ConnectionError('Grobid server is unavailable.') from e

    @retry(delay=random.uniform(1.0, 5.0))
    def parse_file(
            self,
            pdf_file: str | bytes,
            *,
            consolidate_header: str = ConsolidateHeader.ALL_METADATA,
            consolidate_citations: str = ConsolidateCitations.ALL_METADATA,
            consolidate_funders: str = ConsolidateFunders.NO_CONSOLIDATION,
            include_raw_citations: bool = True,
            include_raw_affiliations: bool = False,
            include_raw_copyrights: bool = False,
            segment_sentences: bool = False,
            generate_ids: bool = False,
            start: int = -1,
            end: int = -1
    ) -> tuple[list[ArticleBlock], MarkdownMeta]:
        """使用grobid将PDF文件解析为XML格式返回

        Args:
            pdf_file: 待处理的pdf文件
            consolidate_header: 如何处理论文头部信息，默认处理全部信息
            consolidate_citations: 如何处理论文的引用文献，默认处理全部信息
            consolidate_funders: 关于资助信息的处理方式，默认不提取
            include_raw_citations:
            include_raw_affiliations:
            include_raw_copyrights:
            segment_sentences:
            generate_ids:
            start:
            end:

        Returns:

        """
        with open(pdf_file, 'rb') as f:
            files = {
                'input': (
                    pdf_file,
                    f,
                    'application/pdf',
                    {'Expires': '0'},
                )
            }

            the_data = {
                'consolidateHeader': consolidate_header,
                'consolidateCitations': consolidate_citations,
                'consolidateFunders': consolidate_funders,
                'teiCoordinates': self.coordinates,
                'start': start,
                'end': end,
                'includeRawCitations': '1' if include_raw_citations else '0',
                'includeRawAffiliations': '1' if include_raw_affiliations else '0',
                'includeRawCopyrights': '1' if include_raw_copyrights else '0',
                'segmentSentences': '1' if segment_sentences else '0',
                'generateIDs': '1' if generate_ids else '0'
            }

            response = self.session.post(self.server_url, files=files, data=the_data, timeout=self.timeout)
            if response.status_code != 200:
                raise ResponseError('下载失败')
            else:
                with open('test/temp.xml', 'w', encoding='utf-8') as f:
                    f.write(response.text)

                return self.__parse_xml(response.text)

    def __parse_xml(self, xml_data: str) -> tuple[list[ArticleBlock], MarkdownMeta]:
        """解析XML文件，提取相关信息

        Args:
            xml_data: 原始的grobid输出xml文本

        Returns:
            元组，包含文本块列表和文档meta信息
        """
        soup = BeautifulSoup(xml_data, 'xml')

        # 提取XML中的标题
        title = soup.find('titleStmt').find('title', {'type': 'main'})
        title = ''.join(filter(str.isprintable, title.text.strip())) if title is not None else ''

        # 提取作者信息
        authors = []
        for author in soup.find('sourceDesc').find_all('persName'):
            first_name = author.find('forename', {'type': 'first'})
            first_name = first_name.text.strip() if first_name is not None else ''
            middle_name = author.find('forename', {'type': 'middle'})
            middle_name = middle_name.text.strip() if middle_name is not None else ''
            last_name = author.find('surname')
            last_name = last_name.text.strip() if last_name is not None else ''

            if middle_name != '':
                authors.append(self.__extract_author_name(last_name, f'{first_name} {middle_name}'))
            else:
                authors.append(self.__extract_author_name(last_name, first_name))

        if len(authors) == 0:
            authors.append('')

        # 提取出版年份
        pub_date = soup.find('publicationStmt')
        year_block = pub_date.find('date')
        year = year_block.attrs.get('when') if year_block is not None else ''

        try:
            match len(year):
                case 4:
                    year = int(year)
                case 0:
                    year = -1
                case _:
                    year = int(year[:4])
        except TypeError:
            year = -1

        file_meta = MarkdownMeta(
            title=title,
            author=authors[0],
            year=year,
            source=[]
        )

        doi = soup.find('sourceDesc').find('idno', {'type': 'DOI'})
        if doi:
            file_meta.source.append(
                FileSource(
                    source_url=f'https://doi.org/{doi.text}',
                    source_type=SourceType.WEB
                )
            )

        sections = [ArticleBlock(text=title, text_level=1)]
        # 提取摘要
        abstract_list = soup.find('profileDesc').select('abstract p')
        if len(abstract_list) > 0:
            sections.append(ArticleBlock(text='Abstract', text_level=2))
            for p in abstract_list:
                sections.append(ArticleBlock(text=p.text.strip()))

        # 提取章节信息
        for section in soup.find('body').find_all('div'):
            # 提取章节标题
            if section.find('head') is None:
                continue
            section_title = section.find('head').text.strip()
            title_level = section.find('head').attrs.get('n')
            if title_level:
                matches = re.findall(r'\d', title_level)
                level = len(matches) + 1
            else:
                level = 2

            sections.append(ArticleBlock(text=section_title, text_level=level))

            # 提取章节正文
            for p in section.find_all('p'):
                if p.text:
                    text = ''.join(filter(str.isprintable, p.text.strip()))
                    sections.append(ArticleBlock(text=text))

        return sections, file_meta

    @staticmethod
    def __extract_author_name(surname, given_names) -> str:
        """提取作者的姓名首字母缩写和姓氏

        Args:
            surname: 作者的姓氏
            given_names: 作者的名字，可以包含多个名字

        Returns:
            格式化后的作者姓名，格式为“姓, 名首字母缩写”
        """
        initials = ' '.join([name[0] + '.' for name in given_names.split()])

        return f'{surname}, {initials}'


class PdfLoader(BaseFileLoader):
    """
    Attributes:
        solver: 具体PDF的解析策略
        connector: 具体的解析器连接会话

    Examples:
        若使用Grobid，则先定义解析器，之后将之传入加载器：
        ```python

        gr_setting = get_settings().fileloader.grobid

        with GrobidConnector(gr_setting) as connector:
            loader = PdfLoader(keep_title=True, add_toc=True, solver='grobid', connector=connector)

            pdf_list = glob.glob('test/*.pdf')
            for file in tqdm(pdf_list, total=len(pdf_list)):
                meta, docs = loader.load(file)
                loader.save_md(f'test/md/{Path(file).name.replace(".pdf", ".md")}')
        ```
    """
    solver: Literal['grobid', 'doc2x']
    connector: Any

    def load(
            self, origin_file_path: FilePath, **kwargs
    ) -> tuple[MarkdownMeta, list[Document]]:
        assert self.connector

        if self.solver == 'grobid':
            self.article, self.file_meta = self.connector.parse_file(origin_file_path)
        elif self.solver == 'doc2x':
            raise NotImplementedError

        self.file_meta.source.append(FileSource(source_url=origin_file_path, source_type=SourceType.PDF))
        doc_list = self._article_to_doc(**kwargs)

        return self.file_meta, doc_list
