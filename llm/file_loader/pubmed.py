import gzip
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

from loguru import logger
from pydantic import AnyHttpUrl, FilePath, ValidationError
from tqdm import tqdm

from llm.schemas.pubmed_data import PubMedData


def pubmed_xml_loader(
    origin_file_path: FilePath | AnyHttpUrl, tqdm_position: int = 0
) -> dict[str, PubMedData]:
    """
    从PubMed XML文件中提取文章信息

    Args:
        origin_file_path: PubMed XML文件路径（通常是.gz压缩文件）
        tqdm_position: 用于多进程时控制进度条位置

    Returns:
        dict: 包含提取信息的字典
    """
    result = {}

    try:
        with gzip.open(origin_file_path, 'rb') as f:
            context = ET.iterparse(f, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)

            last_date = date(1970, 1, 1)
            bar = tqdm(
                total=30000,
                desc=f'解析 {Path(origin_file_path).name}',
                position=tqdm_position,
                leave=False,
            )

            for event, element in context:
                if event == 'end' and element.tag == 'PubmedArticle':
                    article_data = {}

                    # 提取PMID
                    pmid_elem = element.find('.//PMID')
                    if pmid_elem is not None:
                        article_data['pmid'] = pmid_elem.text.strip()

                    # 提取文章标题
                    title_elem = element.find('.//ArticleTitle')
                    if title_elem is not None:
                        article_data['title'] = (''.join(title_elem.itertext())).strip()

                    # 提取发表日期
                    pub_date_elem = element.find('.//DateCompleted')
                    if pub_date_elem is None:
                        pub_date_elem = element.find('.//DateRevised')
                    if pub_date_elem is not None:
                        year_elem = pub_date_elem.find('Year')
                        month_elem = pub_date_elem.find('Month')
                        day_elem = pub_date_elem.find('Day')

                        year = int(year_elem.text) if year_elem is not None else last_date.year
                        month = int(month_elem.text) if month_elem is not None else last_date.month
                        day = int(day_elem.text) if day_elem is not None else last_date.day

                        article_date = date(year, month, day)
                        article_data['pub_date'] = article_date
                        last_date = article_date
                    else:
                        # 假设列表内论文的日期是挨着的
                        article_data['pub_date'] = last_date

                    # 提取作者列表
                    authors = []
                    author_list = element.findall('.//AuthorList/Author')
                    for author in author_list:
                        author_data = {}

                        last_name = author.find('LastName')
                        fore_name = author.find('ForeName')
                        if last_name is not None and fore_name is not None:
                            author_name = f'{fore_name.text} {last_name.text}'.strip()
                        elif last_name is not None:
                            author_name = last_name.text.strip()
                        else:
                            continue
                        author_data['name'] = author_name

                        affiliation_elem = author.find('./AffiliationInfo/Affiliation')
                        if affiliation_elem is not None:
                            author_data['affiliation'] = affiliation_elem.text.strip()

                        authors.append(author_data)
                    article_data['author'] = authors

                    # 提取期刊名称和ISSN编号
                    journal_data = {}
                    journal_title_elem = element.find('.//Journal/ISOAbbreviation')
                    if journal_title_elem is None:
                        journal_title_elem = element.find('.//Journal/Title')
                    if journal_title_elem is not None:
                        journal_data['name'] = journal_title_elem.text.strip()

                    issn_elem = element.find('.//Journal/ISSN[@IssnType="Print"]')
                    if issn_elem is not None:
                        journal_data['issn'] = issn_elem.text.strip()
                    else:
                        # 如果没有Print类型的ISSN，尝试获取任何类型的ISSN
                        issn_elem = element.find('.//Journal/ISSN')
                        if issn_elem is not None:
                            journal_data['issn'] = issn_elem.text.strip()
                    article_data['journal'] = journal_data

                    # 提取DOI号
                    doi = element.find(".//ELocationID[@EIdType='doi']")
                    if doi is not None:
                        article_data['doi'] = doi.text.strip()

                    # 提取摘要
                    abstract_elem = element.find('.//Abstract')
                    if abstract_elem is not None:
                        abstract_parts = abstract_elem.findall('./AbstractText')
                        if abstract_parts:
                            abstract = ''
                            for part in abstract_parts:
                                # 获取标签属性
                                label = part.get('Label')

                                # 获取文本内容，包括可能的子元素
                                text = ''
                                if part.text:
                                    text += part.text

                                # 处理可能的子元素（如<i>、<b>等）
                                for child in part:
                                    if child.text:
                                        text += child.text
                                    if child.tail:
                                        text += child.tail

                                # 处理元素的尾部文本
                                if part.tail:
                                    text += part.tail

                                # 组合标签和文本
                                if label:
                                    abstract += f'{label}: {text}\n'
                                else:
                                    abstract += f'{text}\n'

                            article_data['abstract'] = abstract.strip()

                    # 提取关键词列表
                    keywords = []
                    keyword_list = element.findall('.//KeywordList/Keyword')
                    for keyword in keyword_list:
                        if keyword is not None and keyword.text and len(keyword.text) <= 50:
                            keywords.append(keyword.text.replace('\n', ' ').strip())
                    article_data['keywords'] = keywords if keywords else []

                    # 提取引用列表（只提取有PubMed ID的引用）
                    references = []
                    reference_list = element.findall('.//PubmedData/ReferenceList/Reference')
                    for reference in reference_list:
                        pubmed_id = reference.find('./ArticleIdList/ArticleId[@IdType="pubmed"]')
                        if pubmed_id is not None and pubmed_id.text:
                            references.append(pubmed_id.text.strip())
                    article_data['references'] = references if references else []

                    # 提取引用数
                    ref_num_elem = element.find('.//NumberOfReferences')
                    ref_num = int(ref_num_elem.text) if ref_num_elem is not None else 0
                    article_data['reference_num'] = ref_num

                    article_id = article_data['pmid']
                    result[article_id] = PubMedData.model_validate(article_data, strict=True)

                    root.clear()
                    bar.update(1)

            return result

    except ValidationError:
        logger.exception(f'PubMed 数据类型校验错误: {article_data["pmid"]}')
        raise
    except Exception as e:
        logger.exception(f'解析PubMed文件时出错: {str(e)}')
        raise


if __name__ == '__main__':
    for k, v in pubmed_xml_loader('../../test/pubmed25n1274.xml.gz').items():
        print(k)
        print(v.model_dump())
        print(str(v.pub_date))
        break
