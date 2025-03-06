import gzip
import xml.etree.ElementTree as ET

from loguru import logger
from pydantic import AnyHttpUrl, FilePath


def pubmed_xml_loader(origin_file_path: FilePath | AnyHttpUrl) -> dict:
    """
    从PubMed XML文件中提取文章信息

    Args:
        origin_file_path: PubMed XML文件路径（通常是.gz压缩文件）

    Returns:
        dict: 包含提取信息的字典
    """
    result = {}

    try:
        with gzip.open(origin_file_path, 'rb') as f:
            context = ET.iterparse(f, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)

            article_count = 0

            for event, element in context:
                if event == 'end' and element.tag == 'PubmedArticle':
                    article_data = {}

                    # 提取PMID
                    pmid_elem = element.find('.//PMID')
                    if pmid_elem is not None:
                        article_data['pmid'] = pmid_elem.text

                    # 提取文章标题
                    title_elem = element.find('.//ArticleTitle')
                    if title_elem is not None:
                        article_data['title'] = title_elem.text

                    # 提取发表年份
                    pub_date = element.find('.//PubDate/Year')
                    if pub_date is not None:
                        article_data['year'] = int(pub_date.text)

                    # 提取期刊名称
                    journal = element.find('.//Journal/Title')
                    if journal is not None:
                        article_data['journal'] = journal.text

                    # 提取第一作者
                    first_author = element.find('.//AuthorList/Author')
                    if first_author is not None:
                        last_name = first_author.find('LastName')
                        fore_name = first_author.find('ForeName')
                        if last_name is not None and fore_name is not None:
                            article_data['first_author'] = f'{fore_name.text} {last_name.text}'
                        elif last_name is not None:
                            article_data['first_author'] = last_name.text

                    # 提取DOI号
                    doi = element.find(".//ELocationID[@EIdType='doi']")
                    if doi is not None:
                        article_data['doi'] = doi.text

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

                    # 提取引用列表（只提取有PubMed ID的引用）
                    references = []
                    reference_list = element.findall('.//PubmedData/ReferenceList/Reference')
                    for reference in reference_list:
                        pubmed_id = reference.find('./ArticleIdList/ArticleId[@IdType="pubmed"]')
                        if pubmed_id is not None and pubmed_id.text:
                            references.append(pubmed_id.text)

                    if references:
                        article_data['references'] = references

                    if 'doi' in article_data:
                        article_id = article_data.get('pmid', str(article_count))
                        result[article_id] = article_data

                    root.clear()

            return result

    except Exception as e:
        logger.error(f'解析PubMed文件时出错: {str(e)}')
        raise

