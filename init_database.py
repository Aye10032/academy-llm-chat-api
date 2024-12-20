import json
import os
import shutil
import sys
from datetime import datetime

from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger
from sqlmodel import select
from tqdm import tqdm

from app.core.config import get_settings
from app.core.security import get_password_hash
from app.db.session import get_simple_session, create_db_and_tables
from app.models.user import User
from app.schemas.user import UserRole, UserPublic
from llm.core.embedding_core import BgeM3Embeddings
from llm.core.model_core import load_embedding

logger.remove()
handler_id = logger.add(sys.stderr, level="DEBUG")
logger.add('log/init_database.log')


# def init_retriever() -> ParentDocumentRetriever:
#     logger.info('start loading embedding model...')
#     embedding = load_embedding()
#
#     logger.info(f'load collection [{collection_name}], using model {embed_cfg.MODEL}')
#
#     if args.drop_old:
#         doc_store = SqliteDocStore(
#             connection_string=config.get_sqlite_path(collection_name),
#             drop_old=True
#         )
#     else:
#         doc_store = SqliteDocStore(
#             connection_string=config.get_sqlite_path(collection_name)
#         )
#
#     vector_db = Milvus(
#         embedding,
#         collection_name=collection_name,
#         connection_args=milvus_cfg.get_conn_args(),
#         index_params=milvus_cfg.get_collection().index_param,
#         drop_old=True,
#         auto_id=True,
#         enable_dynamic_field=True,
#     )
#     init_doc = Document(page_content=f'This is a collection about {collection_name}',
#                         metadata={
#                             'title': 'About this collection',
#                             'section': 'Abstract',
#                             'author': 'administrator',
#                             'year': datetime.now().year,
#                             'type': -1,
#                             'keywords': 'collection',
#                             'doi': ''
#                         })
#     init_ids = vector_db.add_documents([init_doc])
#     vector_db.delete(init_ids)
#     logger.info('done')
#
#     parent_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=450,
#         chunk_overlap=0,
#         separators=['\n\n', '\n'],
#         keep_separator=False
#     )
#
#     if milvus_cfg.get_collection().language == 'en':
#         child_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=100,
#             chunk_overlap=0,
#             separators=['.', '\n\n', '\n'],
#             keep_separator=False
#         )
#     elif milvus_cfg.get_collection().language == 'zh':
#         child_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=100,
#             chunk_overlap=0,
#             separators=['。', '？', '\n\n', '\n'],
#             keep_separator=False
#         )
#     else:
#         raise Exception(f'error language {milvus_cfg.get_collection().language}')
#
#     retriever = ParentDocumentRetriever(
#         vectorstore=vector_db,
#         docstore=doc_store,
#         child_splitter=child_splitter,
#         parent_splitter=parent_splitter,
#     )
#
#     return retriever
#
#
# def load_md(base_path: str) -> None:
#     """
#     加载markdown文件到检索器中。
#
#     :param base_path: 基础路径，包含年份子目录，每个子目录下包含markdown和xml文件。
#     :return: 无返回值
#     """
#     # 初始化检索器，并添加初始文档
#
#     retriever = init_retriever()
#     now_collection = config.milvus_config.get_collection().collection_name
#     logger.info('start loading file...')
#
#     # 遍历基础路径下的所有文件和子目录
#     for root, dirs, files in os.walk(base_path):
#         # 跳过空目录
#         if len(files) == 0:
#             continue
#
#         # 提取年份信息
#         year = os.path.basename(root)
#         for _file in tqdm(files, total=len(files), desc=f'load file in ({year})'):
#             # 加载并处理markdown文件
#             file_path = os.path.join(config.get_md_path(now_collection), year, _file)
#
#             # 分割markdown文本为多个文档
#             md_docs, reference_data = load_from_md(file_path)
#
#             # 尝试将分割得到的文档添加到检索器
#             try:
#                 retriever.add_documents(md_docs)
#                 with ReferenceStore(config.get_reference_path()) as ref_store:
#                     ref_store.add_reference(reference_data)
#             except Exception as e:
#                 logger.error(f'loading <{_file}> ({year}) fail')
#                 logger.error(e)
#
#     logger.info(f'done')


def init_user():
    setting = get_settings()

    db = get_simple_session()
    create_db_and_tables()

    statement = select(User).where(User.email == setting.server.INIT_USER)
    test_user = db.exec(statement).first()
    if test_user:
        logger.warning("User already exist")
        logger.warning(UserPublic(**test_user.model_dump()))
        return

    test_user: User = User(
        email=setting.server.INIT_USER,
        username="Admin",
        hashed_password=get_password_hash(setting.server.INIT_PASSWORD),
        is_active=True,
        role=UserRole.admin
    )

    db.add(test_user)
    db.commit()
    logger.info("Done")

    db.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='database init')
    parser.add_argument(
        '--collection',
        '-C',
        nargs='?',
        const=-1,
        type=int,
        help='Initialize a specific collection, starting from 0.'
    )
    parser.add_argument(
        '--user',
        '-U',
        action='store_true',
        help='Init default admin profile'
    )
    parser.add_argument(
        '--force',
        '-F',
        action='store_true',
        help='Force override of existing configurations'
    )
    parser.add_argument(
        '--drop_old',
        '-D',
        action='store_true',
        help='Whether to delete the original reference database'
    )
    args = parser.parse_args()

    if args.user:
        logger.info('Create admin profile...')
        init_user()

    # if args.auto_create:
    #     setting = get_settings()
    #
    #     init_user(setting.server)
    #
    # if args.drop_old:
    #     with ReferenceStore(config.get_reference_path()) as _store:
    #         _store.drop_old()
    #         logger.info('drop old database.')
    #
    # if args.collection is not None:
    #     if args.collection == -1:
    #         for i in range(len(config.milvus_config.collections)):
    #             logger.info(f'Start init collection {i}')
    #             config.set_collection(i)
    #             load_md(config.get_md_path(config.milvus_config.get_collection().collection_name))
    #     else:
    #         if args.collection >= len(config.milvus_config.collections) or args.collection < -1:
    #             logger.error(f'collection index {args.collection} out of range')
    #             exit(1)
    #         else:
    #             config.set_collection(args.collection)
    #             logger.info(f'Only init collection {args.collection}')
    #             load_md(config.get_md_path(config.milvus_config.get_collection().collection_name))
