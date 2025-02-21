import argparse
import glob
import os.path
import shutil
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from loguru import logger
from tqdm import tqdm
from urllib3.exceptions import ResponseError

from app.core.config import get_settings
from app.core.security import get_password_hash
from app.crud.knowledge_base import (
    insert_knowledge_base,
    KBExistError,
    update_knowledge_base,
    get_knowledge_base_by_name,
    delete_by_name,
)
from app.crud.user import insert_user, UserExistError
from app.db.session import get_simple_session, create_db_and_tables
from app.models import UserTable, KnowledgeBaseTable
from app.schemas.knowledge_base import KnowledgeBaseUpdate
from app.schemas.user import UserRole
from app.utils.validator import validate_input, simple_char_valid
from llm.core.model import load_embedding
from llm.file_loader import MarkdownLoader
from llm.file_loader.loader import FileLoadError
from llm.file_loader.pdf import PdfLoader, GrobidConnector, DOINotFoundError
from llm.rag.retriever import insert_chain
from llm.rag.storage import create_vector_db, get_doc_db, fix_null_fields, get_vector_db
from llm.schemas.markdown import SourceType, FileSource

logger.remove()
handler_id = logger.add(sys.stderr, level='DEBUG')
logger.add('log/init_database.log')


def init_user(email: str, password: str):
    test_user = UserTable(
        email=email,
        username='Admin',
        hashed_password=get_password_hash(password),
        is_active=True,
        role=UserRole.ADMIN,
    )

    session = get_simple_session()
    logger.info('从配置文件创建默认管理员账户...')
    try:
        insert_user(session, test_user)
        logger.info(
            '创建完毕。请注意，当存在手动注册的其他管理员账户后，此账号将被禁用。'
        )
    except UserExistError:
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
    user_input = input(
        '\033[33m输入文献摘要段落的检查关键字（默认为abstract）： \033[0m'
    )
    return user_input if user_input else 'abstract'


def init_knowledge_base(file_path: str, output_path: str, drop_old: bool):
    logger.info(f'覆盖:{drop_old}')
    session = get_simple_session()

    # 创建知识库相关数据表
    collection_name = _get_collection_name()
    if get_knowledge_base_by_name(session, collection_name) and not drop_old:
        abs_key = _get_collection_abstract_keyword()
        collection_lang = _get_collection_lang()
        collection_ext = _get_collection_ext()
        uid = get_knowledge_base_by_name(session, collection_name).uid

        now_time = datetime.now()
        knowledge_base = KnowledgeBaseUpdate(last_update=now_time)
        update_knowledge_base(session, uid, knowledge_base)

        # 初始化向量数据库
        embedding_model = load_embedding()

        logger.info('加载向量数据库...')
        vector_db = get_vector_db(
            table_name=collection_name,
            embedding_model=embedding_model,
            db_name='llm_chat',
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
            delete_by_name(session, collection_name)

        try:
            insert_knowledge_base(session, knowledge_base)
        except KBExistError:
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

    if not os.path.exists(file_path):
        logger.error(f'路径 {file_path} 不存在！')
        exit(0)

    doc_db = get_doc_db(collection_name, drop_old=drop_old)
    retriever = insert_chain(vector_db, doc_db, collection_lang)

    if drop_old:
        shutil.rmtree(os.path.join(output_path, collection_name))

    os.makedirs('temp', exist_ok=True)

    if collection_ext == 'md':
        markdown_list = glob.glob(f'{file_path.rstrip("/")}/*.md')
        md_path = os.path.join(output_path, collection_name, 'markdown')
        os.makedirs(md_path, exist_ok=True)

        md_loader = MarkdownLoader(keep_title=False, abstract_key=abs_key)
        logger.info('建立文档索引')
        for file in tqdm(markdown_list, total=len(markdown_list)):
            md_file = os.path.join(md_path, Path(file).name)
            if Path(md_file).exists():
                continue

            _, docs = md_loader.load(file)
            md_loader.save_md(md_file)

            # TODO 临时修复zilliz无法处理空值的问题
            if get_settings().retriever.knowledge_base.milvus.SECURE:
                docs = fix_null_fields(docs)

            retriever.add_documents(docs)
    elif collection_ext == 'pdf':
        pdf_list = glob.glob(f'{file_path.rstrip("/")}/*.pdf')
        pdf_path = os.path.join(output_path, collection_name, 'pdf')
        md_path = os.path.join(output_path, collection_name, 'markdown')
        os.makedirs(pdf_path, exist_ok=True)
        os.makedirs(md_path, exist_ok=True)
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
                pdf_file = os.path.join(pdf_path, Path(file).name)
                if Path(pdf_file).exists():
                    time.sleep(0.1)
                    continue

                shutil.copyfile(file, pdf_file)

                try:
                    _, docs = loader.load(pdf_file)

                    md_file = os.path.join(
                        md_path, Path(file).name.replace('.pdf', '.md')
                    )
                    loader.save_md(md_file)

                    # TODO 临时修复zilliz无法处理空值的问题
                    if get_settings().retriever.knowledge_base.milvus.SECURE:
                        docs = fix_null_fields(docs)
                    retriever.add_documents(docs)
                except DOINotFoundError:
                    os.remove(pdf_file)
                    logger.error(f'文件{file}未识别到doi，请手动处理')
                except FileLoadError:
                    os.remove(pdf_file)
                    logger.error(f'文件{file}解析失败，请手动处理')
                except ResponseError:
                    os.remove(pdf_file)
                    logger.error(f'文件{file}下载失败，请手动处理')


def load_args(args: Namespace):
    setting = get_settings()
    create_db_and_tables()

    if args.user:
        init_user(setting.server.INIT_USER, setting.server.INIT_PASSWORD)

    if origin_path := args.knowledge_base:
        init_knowledge_base(
            origin_path,
            setting.retriever.knowledge_base.STORE_PATH,
            args.drop_old,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='database init')
    parser.add_argument('--user', '-u', action='store_true', help='创建默认管理员账户')
    parser.add_argument(
        '--knowledge_base',
        '-kb',
        nargs='?',
        const=-1,
        type=str,
        default='',
        help='创建知识库。若传入待建库文件所在路径，则会进行知识库文件的初始化。目前仅支持一次性处理单一类型的文件',
    )
    parser.add_argument(
        '--drop_old', '-d', action='store_true', help='是否直接覆盖原有数据'
    )
    load_args(parser.parse_args())
