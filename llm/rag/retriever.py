from typing import Optional, Any, Literal

from langchain.retrievers import ParentDocumentRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus
from langchain_milvus.vectorstores import milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from llm.rag.storage import SqliteDocStore

retriever_cfg = get_settings().retriever


def get_vector_db(
        table_name: str,
        embedding_model: Embeddings,
        *,
        db_name: str = 'default',
        index_params: Optional[dict[str, Any]] = None
) -> Milvus:
    """
    Retrieve a Milvus vector database instance.

    Args:
        table_name (str): The name of the table to use in the vector database.
        embedding_model (Embeddings): The embedding model to use for vectorization.
        db_name (str): The name of the database. Defaults to 'default'.
        index_params (Optional[dict[str, Any]]): Parameters for indexing. Defaults to None.

    Returns:
        Milvus: An instance of the Milvus vector database.
    """
    vector_db: milvus = Milvus(
        embedding_model,
        collection_name=table_name,
        connection_args=retriever_cfg.knowledge_base.milvus.get_conn_args(db_name),
        index_params=index_params,
        search_params={'metric_type': 'L2', 'params': {'ef': 10}},
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

    if language == 'en':
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,
            separators=['.', '\r\n', '\n'],
            keep_separator=False
        )
    elif language == 'zh':
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=0,
            separators=['。', '\r\n', '\n'],
            keep_separator=False
        )

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever
