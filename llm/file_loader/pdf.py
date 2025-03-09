import random
import re
from enum import StrEnum
from typing import Any, Literal

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from loguru import logger
from pydantic import FilePath
from requests import RequestException
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from urllib3.exceptions import ResponseError

from app.core.config import GrobidSetting, get_settings
from app.utils.network import retry
from llm.file_loader.loader import BaseFileLoader
from llm.schemas import ArticleBlock, MarkdownMeta
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


class DOINotFoundError(Exception):
    pass


@retry(delay=random.uniform(2.0, 5.0))
def get_paper_info(pmid: str, silent: bool = True) -> tuple[list[ArticleBlock], MarkdownMeta]:
    """根据pubmed id获取文献信息

    Args:
        pmid: PubMed ID
        silent: 是否输出日志信息

    Returns:

    """

    if not silent:
        logger.info(f'request PMID:{pmid}')

    url = (
        f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml'
    )
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0'
    }

    if get_settings().server.network.USE_PROXY:
        proxies = {
            'http': get_settings().server.network.PROXY,
            'https': get_settings().server.network.PROXY,
        }
        response = requests.request('GET', url, headers=headers, proxies=proxies, timeout=10)
    else:
        response = requests.request('GET', url, headers=headers, timeout=10)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'xml')

        title = soup.find('Article').find('ArticleTitle').text if soup.find('Article') else None
        year = soup.find('Article').find('JournalIssue').find('PubDate').find('Year').text

        author = ''
        if author_block := soup.find('Author'):
            last_name = author_block.find('LastName').text if author_block.find('LastName') else ''
            initials = author_block.find('Initials').text if author_block.find('Initials') else ''
            author = f'{last_name}, {initials}'

        abstract = soup.find('AbstractText').text if soup.find('AbstractText') else None

        doi_block = soup.find('ArticleIdList').find('ArticleId', {'IdType': 'doi'})
        if doi_block:
            doi = doi_block.text
        else:
            doi = None
            logger.warning('DOI not found')

        paper_info = MarkdownMeta(
            title=title,
            author=author,
            year=int(year),
            source=[
                FileSource(source_url=f'https://doi.org/{doi}', source_type=SourceType.WEB),
                FileSource(
                    source_url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                    source_type=SourceType.PUBMED,
                ),
            ],
        )

        section_list = [
            ArticleBlock(text=title, text_level=1),
            ArticleBlock(text='Abstract', text_level=2),
            ArticleBlock(text=abstract),
        ]
        return section_list, paper_info
    else:
        # 请求失败时抛出异常
        raise ResponseError('下载请求失败')


@retry(delay=random.uniform(2.0, 5.0))
def get_info_by_doi(doi: str, silent: bool = True) -> tuple[list[ArticleBlock], MarkdownMeta]:
    """通过DOI号补全文献信息

    Args:
        doi: 待查询文献的DOI号
        silent: 是否输出日志信息

    Returns:

    """
    if not silent:
        logger.info(f'search paper: {doi}')

    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={doi}[doi]&retmode=xml'

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0'
    }

    if get_settings().server.network.USE_PROXY:
        proxies = {
            'http': get_settings().server.network.PROXY,
            'https': get_settings().server.network.PROXY,
        }
        response = requests.request('GET', url, headers=headers, proxies=proxies, timeout=10)
    else:
        response = requests.request('GET', url, headers=headers, timeout=10)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'xml')

        count = int(soup.find('Count').text) if soup.find('Count') else 0

        if count > 0:
            pmid = soup.find('IdList').find_all('Id')[0].text
            return get_paper_info(pmid)
        else:
            raise DOINotFoundError
    else:
        raise ResponseError


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

        retries = Retry(total=5, backoff_factor=5, status_forcelist=[500, 502, 503, 504, 300])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.session.headers.update({'Accept': 'application/xml'})

        if get_settings().server.network.USE_PROXY:
            k, v = get_settings().server.network.PROXY.split('://')
            self.session.proxies.update({k: v})

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
        end: int = -1,
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
                'generateIDs': '1' if generate_ids else '0',
            }

            response = self.session.post(
                self.server_url, files=files, data=the_data, timeout=self.timeout
            )
            if response.status_code != 200:
                raise ResponseError('下载失败')
            else:
                with open('temp/temp.xml', 'w', encoding='utf-8') as xml_f:
                    xml_f.write(response.text)

                return self.__parse_xml(response.text)

    def __parse_xml(self, xml_data: str) -> tuple[list[ArticleBlock], MarkdownMeta]:
        """解析XML文件，提取相关信息

        Args:
            xml_data: 原始的grobid输出xml文本

        Returns:
            元组，包含文本块列表和文档meta信息
        """
        soup = BeautifulSoup(xml_data, 'xml')

        doi = soup.find('sourceDesc').find('idno', {'type': 'DOI'})
        if doi:
            sections, file_meta = get_info_by_doi(doi.text)
        else:
            raise DOINotFoundError

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

    def load(self, origin_file_path: FilePath, **kwargs) -> tuple[MarkdownMeta, list[Document]]:
        assert self.connector

        if self.solver == 'grobid':
            self.article, self.file_meta = self.connector.parse_file(origin_file_path)
        elif self.solver == 'doc2x':
            raise NotImplementedError

        self.file_meta.source.append(
            FileSource(source_url=origin_file_path, source_type=SourceType.PDF)
        )
        doc_list = self._article_to_doc(**kwargs)

        return self.file_meta, doc_list
