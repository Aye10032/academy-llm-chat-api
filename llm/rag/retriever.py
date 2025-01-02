from typing import Optional, Any, Literal

from langchain.retrievers import ParentDocumentRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus
from langchain_milvus.vectorstores import milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import DataType, MilvusClient
from pymilvus.orm.types import infer_dtype_bydata

from app.core.config import get_settings
from llm.rag.storage import SqliteDocStore

retriever_cfg = get_settings().retriever


def create_vector_db(
        table_name: str,
        embedding_model: Embeddings,
        *,
        db_name: str = 'default',
        index_params: Optional[dict[str, Any]] = None
) -> Milvus:
    """初始化Milvus数据库

    Args:
        table_name (str): 数据表名称
        embedding_model (Embeddings): 向量查询所使用的嵌入模型
        db_name (str): 连接的数据库，默认为 'default'.
        index_params (Optional[dict[str, Any]]): （可选参数）自定义索引算法参数

    Returns:
        Milvus: langchain格式的Milvus数据库对象
    """
    vector_field_embeddings = embedding_model.embed_documents(['test'])
    dim = len(vector_field_embeddings[0])

    client = MilvusClient(**retriever_cfg.knowledge_base.milvus.get_conn_args(db_name))

    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
    )
    schema.add_field('title', DataType.VARCHAR, max_length=65535)
    schema.add_field('section', DataType.VARCHAR, max_length=65535, default_value='')
    schema.add_field('section_3', DataType.VARCHAR, max_length=65535, default_value='')
    schema.add_field('section_4', DataType.VARCHAR, max_length=65535, default_value='')
    schema.add_field('section_5', DataType.VARCHAR, max_length=65535, default_value='')
    schema.add_field('section_6', DataType.VARCHAR, max_length=65535, default_value='')
    schema.add_field('author', DataType.VARCHAR, max_length=65535)
    schema.add_field('year', DataType.INT64)
    schema.add_field('type', DataType.VARCHAR, max_length=65535)
    schema.add_field('source', DataType.VARCHAR, max_length=65535)
    schema.add_field('source_type', DataType.INT8)
    schema.add_field('doc_id', DataType.VARCHAR, max_length=65535)
    schema.add_field('text', DataType.VARCHAR, max_length=65535)
    schema.add_field('pk', DataType.INT64, is_primary=True)
    schema.add_field('vector', infer_dtype_bydata(vector_field_embeddings[0]), dim=dim)

    client.create_collection(
        collection_name=table_name,
        dimension=dim,
        primary_field_name='pk',
        vector_field_name='vector',
        metric_type='L2',
        schema=schema,
        auto_id=True,
        index_params=index_params
    )
    client.close()

    vector_db = get_vector_db(table_name, embedding_model, db_name=db_name)

    return vector_db


def get_vector_db(
        table_name: str,
        embedding_model: Embeddings,
        *,
        db_name: str = 'default',
        ef: int = 10
) -> Milvus:
    """返回Milvus向量数据库对象

    Args:
        table_name (str): 数据表名称
        embedding_model (Embeddings): 向量查询所使用的嵌入模型
        db_name (str): 连接的数据库，默认为 'default'.
        ef: 相似性搜索返回的结果数量

    Returns:
        Milvus: langchain格式的Milvus数据库对象
    """
    vector_db: milvus = Milvus(
        embedding_model,
        collection_name=table_name,
        connection_args=retriever_cfg.knowledge_base.milvus.get_conn_args(db_name),
        search_params={'metric_type': 'L2', 'params': {'ef': ef}},
        auto_id=True,
        enable_dynamic_field=False,
    )

    return vector_db


def get_doc_db(table_name: str) -> BaseStore:
    doc_store = SqliteDocStore(
        connection_string=retriever_cfg.knowledge_base.DOC_URL,
        table_name=table_name,
        drop_old=True
    )

    return doc_store


def insert_chain(
        vector_store: VectorStore,
        doc_store: BaseStore,
        language: str = Literal['en', 'zh']
) -> ParentDocumentRetriever:
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=['\n'],
        keep_separator=False
    )

    if language == 'zh':
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=0,
            separators=['。', '\r\n', '\n'],
            keep_separator=False
        )
    else:
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,
            separators=['. ', '\r\n', '\n'],
            keep_separator=False
        )

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever
