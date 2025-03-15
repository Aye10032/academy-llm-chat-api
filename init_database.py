import asyncio
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import click
from langchain_core.documents import Document
from loguru import logger
from tqdm.asyncio import tqdm
from urllib3.exceptions import ResponseError

import app.crud.knowledge_base as kb_crud
import app.crud.user as user_crud
from app.core.config import get_settings
from app.core.security import get_password_hash
from app.db.session import create_db_and_tables, get_simple_session
from app.models import KnowledgeBaseTable, UserTable
from app.schemas.knowledge_base import KnowledgeBaseUpdate
from app.schemas.user import UserRole
from app.utils.ftp import FTPClient, download_http_files
from app.utils.md5 import verify_md5
from app.utils.validator import simple_char_valid, validate_input
from llm.core.model import load_embedding
from llm.file_loader.loader import FileLoadError
from llm.file_loader.markdown import MarkdownLoader
from llm.file_loader.pdf import DOINotFoundError, GrobidConnector, PdfLoader
from llm.file_loader.pubmed import pubmed_xml_loader
from llm.rag.pubmed_graph import init_pubmed_graph, insert_paper
from llm.rag.retriever import insert_chain
from llm.rag.storage import create_vector_db, fix_null_fields, get_doc_db, get_vector_db
from llm.schemas.markdown import FileSource, SourceType

logger.remove()
handler_id = logger.add(sys.stderr, level='DEBUG')
logger.add('log/init_database.log')


@click.group(help='database init')
@click.option('--drop_old', '-d', is_flag=True, help='是否直接覆盖原有数据', default=False)
@click.pass_context
def cli(ctx, drop_old: bool):
    ctx.obj = {'drop_old': drop_old}


@cli.command(name='user', help='创建默认管理员账户')
@click.pass_context
def init_user(ctx):
    create_db_and_tables()

    setting = get_settings()

    test_user = UserTable(
        email=setting.auth.INIT_USER,
        username='Admin',
        hashed_password=get_password_hash(setting.auth.INIT_PASSWORD),
        is_active=True,
        role=UserRole.ADMIN,
    )

    session = get_simple_session()
    logger.info('从配置文件创建默认管理员账户...')
    try:
        user_crud.insert(session, test_user)
        logger.info('创建完毕。请注意，当存在手动注册的其他管理员账户后，此账号将被禁用。')
    except user_crud.UserExistError:
        logger.error('此邮箱已存在')
    finally:
        session.close()


@validate_input(simple_char_valid, '知识库名称只能包含英文字母、数字和下划线')
def _get_collection_name() -> str:
    return input('\033[33m输入知识库名称（仅包含英文大小写、下划线和数字）: \033[0m')


@validate_input(lambda x: len(x) > 0, '标题不能为空')
def _get_collection_title() -> str:
    return input('\033[33m输入知识库的显示标题： \033[0m')


@validate_input(lambda x: x in ['en', 'zh'], '只能是en或zh')
def _get_collection_lang() -> str:
    return input('\033[33m输入知识库文件的语言： \033[0m')


@validate_input(lambda x: x in ['md', 'pdf'], '只能是md、pdf中的一种')
def _get_collection_ext() -> str:
    user_input = input('\033[33m输入本次建库文件的类型（默认为md）： \033[0m')
    return user_input if user_input else 'md'


def _get_collection_abstract_keyword() -> str:
    user_input = input('\033[33m输入文献摘要段落的检查关键字（默认为abstract）： \033[0m')
    return user_input if user_input else 'abstract'


@cli.command(name='knowledge_base', help='创建知识库')
@click.argument('path', default='', required=False)
@click.pass_context
def init_knowledge_base(ctx, path: str):
    create_db_and_tables()

    drop_old = ctx.obj['drop_old']
    output_path = Path(get_settings().knowledge_base.STORE_PATH)
    file_path = Path(path)

    logger.info(f'覆盖:{drop_old}')
    session = get_simple_session()

    # 创建知识库相关数据表
    collection_name = _get_collection_name()
    if kb_crud.get_by_name(session, collection_name) and not drop_old:
        abs_key = _get_collection_abstract_keyword()
        collection_lang = _get_collection_lang()
        collection_ext = _get_collection_ext()
        uid = kb_crud.get_by_name(session, collection_name).uid

        now_time = datetime.now()
        knowledge_base = KnowledgeBaseUpdate(last_update=now_time)
        kb_crud.update(session, uid, knowledge_base)

        # 初始化向量数据库
        embedding_model = load_embedding()

        logger.info('加载向量数据库...')
        vector_db = get_vector_db(
            table_name=collection_name, embedding_model=embedding_model, db_name='llm_chat'
        )
    else:
        collection_title = _get_collection_title()
        collection_desc = input('\033[33m输入知识库描述： \033[0m')
        abs_key = _get_collection_abstract_keyword()
        collection_lang = _get_collection_lang()
        collection_ext = _get_collection_ext()
        now_time = datetime.now()

        knowledge_base = KnowledgeBaseTable(
            uid=str(uuid4()),
            table_name=collection_name,
            table_title=collection_title,
            description=collection_desc,
            create_time=now_time,
            last_update=now_time,
        )

        logger.info('创建知识库记录...')
        if drop_old:
            kb_crud.delete_by_name(session, collection_name)

        try:
            kb_crud.insert(session, knowledge_base)
        except kb_crud.KBExistError:
            logger.error('已经存在同名的知识库')
            exit(0)
        finally:
            session.close()

        # 初始化向量数据库
        embedding_model = load_embedding()

        logger.info('初始化向量数据库...')
        vector_db = create_vector_db(
            table_name=collection_name,
            embedding_model=embedding_model,
            db_name='llm_chat',
            drop_old=drop_old,
        )

    if not file_path.exists():
        logger.error(f'路径 {file_path} 不存在！')
        exit(0)

    doc_db = get_doc_db(collection_name, drop_old=drop_old)
    retriever = insert_chain(vector_db, doc_db, collection_lang)

    if drop_old:
        shutil.rmtree(output_path / collection_name)

    # os.makedirs('temp', exist_ok=True)

    if collection_ext == 'md':
        markdown_list = list(file_path.glob('*.md'))
        md_path = output_path / collection_name / 'markdown'
        md_path.mkdir(parents=True, exist_ok=True)

        md_loader = MarkdownLoader(keep_title=False, abstract_key=abs_key)
        logger.info('建立文档索引')
        for file in tqdm(markdown_list, total=len(markdown_list)):
            md_file = md_path / file.name
            if md_file.exists():
                continue

            _, docs = md_loader.load(file)
            md_loader.save_md(md_file)

            # TODO 临时修复zilliz无法处理空值的问题
            if get_settings().knowledge_base.milvus.SECURE:
                docs = fix_null_fields(docs)

            retriever.add_documents(docs)
    elif collection_ext == 'pdf':
        pdf_list = list(file_path.glob('*.pdf'))
        pdf_path = output_path / collection_name / 'pdf'
        md_path = output_path / collection_name / 'markdown'

        pdf_path.mkdir(parents=True, exist_ok=True)
        md_path.mkdir(parents=True, exist_ok=True)

        gr_setting = get_settings().fileloader.grobid

        with GrobidConnector(gr_setting) as connector:
            loader = PdfLoader(
                keep_title=False,
                add_toc=True,
                solver='grobid',
                connector=connector,
                abstract_key=abs_key,
            )
            logger.info('建立文档索引')
            for file in tqdm(pdf_list, total=len(pdf_list)):
                pdf_file = pdf_path / file.name
                if pdf_file.exists():
                    time.sleep(0.1)
                    continue

                shutil.copyfile(file, pdf_file)

                try:
                    _, docs = loader.load(pdf_file)

                    md_file = md_path / f'{file.stem}.md'
                    loader.save_md(md_file)

                    # TODO 临时修复zilliz无法处理空值的问题
                    if get_settings().knowledge_base.milvus.SECURE:
                        docs = fix_null_fields(docs)
                    retriever.add_documents(docs)
                except DOINotFoundError:
                    pdf_file.unlink(missing_ok=True)
                    logger.error(f'文件{file}未识别到doi，请手动处理')
                except FileLoadError:
                    pdf_file.unlink(missing_ok=True)
                    logger.error(f'文件{file}解析失败，请手动处理')
                except ResponseError:
                    pdf_file.unlink(missing_ok=True)
                    logger.error(f'文件{file}下载失败，请手动处理')


@cli.command(name='pubmed', help='初始化PubMed data知识库')
@click.option('--concurrency', '-c', type=int, default=1, help='最大并发数')
@click.option('--db_name', '-n', type=str, default='pubmed', help='数据库名称，默认为pubmed')
@click.pass_context
def init_pubmed_db(ctx, concurrency: int, db_name: str):
    create_db_and_tables()

    drop_old = ctx.obj['drop_old']

    root_path = Path(get_settings().knowledge_base.STORE_PATH) / db_name
    local_path = Path(root_path) / 'baseline'

    if drop_old and local_path.exists():
        shutil.rmtree(local_path)

    local_path.mkdir(parents=True, exist_ok=True)

    # 获取远端文件列表
    with FTPClient(host='ftp.ncbi.nlm.nih.gov', username='anonymous', password='') as ftp:
        files = ftp.list_files('/pubmed/baseline')

    exist_files = []

    for file_path in local_path.iterdir():
        if file_path.is_file():
            if file_path.stat().st_size == 0:
                file_path.unlink(missing_ok=True)
            else:
                exist_files.append(file_path.name)

    if not drop_old and exist_files:
        download_files = list(set(files) - set(exist_files))
    else:
        download_files = files

    async def download_func():
        await download_http_files(
            base_url='https://ftp.ncbi.nlm.nih.gov',
            remote_path='/pubmed/baseline',
            file_list=download_files,
            local_dir=local_path,
            max_concurrency=concurrency,
        )

    if download_files:
        asyncio.run(download_func())

    # 校验 MD5
    gz_list = list(local_path.glob('*.gz'))
    fail_count = 0
    for gz_file in tqdm(gz_list, total=len(gz_list), desc='MD5校验'):
        md5_file = local_path / f'{gz_file.name}.md5'
        if not verify_md5(gz_file, md5_file):
            gz_file.unlink(missing_ok=True)
            md5_file.unlink(missing_ok=True)
            fail_count += 1

    if fail_count > 0:
        logger.error(f'有 {fail_count} 个文件MD5校验失败，已删除。请重新运行此命令以下载')
        exit(0)
    else:
        logger.info('MD5校验通过')

    # 解析数据库
    session = get_simple_session()
    now_time = datetime.now()
    if drop_old:
        kb_crud.delete_by_name(session, db_name)
        init_pubmed_graph(True)

    embedding_model = load_embedding()
    if now_kb := kb_crud.get_by_name(session, db_name):
        kb_crud.update(session, now_kb.uid, KnowledgeBaseUpdate(last_update=now_time))
        logger.info('加载向量数据库...')
        vector_db = get_vector_db(
            table_name=db_name, embedding_model=embedding_model, db_name='llm_chat'
        )
    else:
        logger.info('创建知识库记录...')
        knowledge_base = KnowledgeBaseTable(
            uid=str(uuid4()),
            table_name=db_name,
            table_title='Pubmed文献知识库',
            description='Pubmed存储的文献归档，包括生物、医学方面的大量文献信息',
            create_time=now_time,
            last_update=now_time,
        )
        kb_crud.insert(session, knowledge_base)

        logger.info('初始化向量数据库...')
        vector_db = create_vector_db(
            table_name=db_name,
            embedding_model=embedding_model,
            db_name='llm_chat',
            drop_old=drop_old,
        )

        logger.info('初始化图数据库...')
        init_pubmed_graph(False)

    doc_db = get_doc_db(db_name, drop_old=drop_old)
    retriever = insert_chain(vector_db, doc_db, 'en')

    for gz_file in tqdm(gz_list, total=len(gz_list), desc='解析归档文件', position=0):
        result = pubmed_xml_loader(gz_file, 1)
        doc_list = []
        for pmid, pubmed_data in tqdm(
            result.items(), total=len(result.items()), desc='建立图索引', position=1
        ):
            insert_paper(pubmed_data)

            file_uid = str(uuid4())

            if pubmed_data.abstract:
                first_author = pubmed_data.author[0].name if pubmed_data.author else 'unknown'
                source = [
                    FileSource(
                        source_url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                        source_type=SourceType.PUBMED,
                    )
                ]
                if pubmed_data.doi:
                    source.extend(
                        [
                            FileSource(
                                source_url=f'https://doi.org/{pubmed_data.doi}',
                                source_type=SourceType.WEB,
                            )
                        ]
                    )
                abstract_doc = Document(
                    page_content=pubmed_data.abstract,
                    metadata={
                        'title': pubmed_data.title,
                        'section': 'Abstract',
                        'author': first_author,
                        'year': pubmed_data.pub_date.year,
                        'type': 'abstract',
                        'source': source,
                        'file_id': file_uid,
                    },
                )
                doc_list.append(abstract_doc)

        retriever.add_documents(doc_list)
        doc_list.clear()


if __name__ == '__main__':
    cli()
