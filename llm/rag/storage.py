import sqlite3
from json import JSONDecodeError
from typing import Any, Iterator, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.load import dumps, loads
from langchain_core.stores import BaseStore
from langchain_milvus import Milvus
from langchain_milvus.vectorstores import milvus
from loguru import logger
from pymilvus import MilvusClient, DataType
from pymilvus.orm.types import infer_dtype_bydata

from app.core.config import get_settings

retriever_cfg = get_settings().retriever


class SqliteDocStore(BaseStore[str, Document]):
    """SQLite-based document store for persisting Document objects.

    Attributes:
        connection_string (str): SQLite database file path.
        table_name (str): Name of the table to store documents.
        drop_old (bool): Whether to drop existing table on initialization.
        engine_args (dict): Additional arguments for SQLite connection.
        iterator_window_size (int): Batch size for iterating through documents.

    Examples:

        .. code-block:: python

            from langchain_core.documents import Document

            from llm.rag.storage import SqliteDocStore

            # Initialize store
            store = SqliteDocStore('docs.db', 'documents')

            # Create sample documents
            doc1 = Document(page_content="Hello", metadata={"source": "doc1"})
            doc2 = Document(page_content="World", metadata={"source": "doc2"})

            # Store documents
            store.mset([('doc1', doc1), ('doc2', doc2)])

            # Retrieve documents
            docs = store.mget(['doc1', 'doc2'])
            # [Document(page_content="Hello"...), Document(page_content="World"...)]

            # Delete documents
            store.mdelete(['doc1'])

            # List all document keys
            list(store.yield_keys())
            # ['doc2']

            # List documents with prefix
            list(store.yield_keys(prefix='doc'))
            # ['doc2']
    """

    def __init__(
            self,
            connection_string: str,
            table_name: str,
            drop_old: bool = False,
            connection: Optional[sqlite3.connect] = None,
            engine_args: Optional[dict[str, Any]] = None,
            iterator_window_size: int = 500
    ) -> None:
        self.connection_string = connection_string
        self.table_name = table_name
        self.drop_old = drop_old
        self.engine_args = engine_args or {}
        self.iterator_window_size = iterator_window_size

        self._conn = connection if connection else self.__connect()
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.drop_old:
            self.__delete_table()
        self.__create_tables_if_not_exists()

    def __connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.connection_string, **self.engine_args, check_same_thread=False)
        return conn

    def __create_tables_if_not_exists(self) -> None:
        cur = self._conn.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is None:
            stmt = f"""CREATE TABLE {self.table_name}
                    (
                        content TEXT,
                        doc_id TEXT
                    );
                    """
            cur.execute(stmt)
            self._conn.commit()
            logger.info(f'Create table {self.table_name}')

        cur.close()

    def __delete_table(self):
        cur = self._conn.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is not None:
            stmt = f"DROP table {self.table_name}"
            cur.execute(stmt)
            self._conn.commit()

        cur.close()

    def __del__(self) -> None:
        if self._conn:
            self._conn.close()

    @staticmethod
    @logger.catch
    def __serialize_value(obj: Document) -> str:
        try:
            return dumps(obj)
        except TypeError as e:
            logger.error(e)

    @staticmethod
    @logger.catch
    def __deserialize_value(obj: str) -> Document:
        try:
            return loads(obj)
        except JSONDecodeError as e:
            logger.error(e)
        except TypeError as e:
            logger.error(e)

    def mget(self, keys: Sequence[str]) -> list[Optional[Document]]:
        cur = self._conn.cursor()
        query = f"""
        SELECT content, doc_id 
        FROM {self.table_name} 
        WHERE doc_id  IN ({','.join(['?'] * len(keys))})
        """

        cur.execute(query, keys)
        items = cur.fetchall()
        cur.close()

        ordered_values = {key: type[Document] for key in keys}
        for item in items:
            v = item[0]
            val: Document = self.__deserialize_value(v)
            k = item[1]
            val.metadata['doc_id'] = k
            ordered_values[k] = val

        return [ordered_values[key] for key in keys]

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        cur = self._conn.cursor()
        data = []
        for doc_id, item in key_value_pairs:
            content = self.__serialize_value(item)
            data.append((content, doc_id))

        cur.executemany(f"INSERT INTO {self.table_name} VALUES(?, ?)", data)
        self._conn.commit()
        cur.close()

    def mdelete(self, keys: Sequence[str]) -> None:
        cur = self._conn.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is None:
            raise ValueError('Collection not found')
        if keys is not None:
            stmt = f"DELETE FROM {self.table_name} WHERE doc_id IN ({','.join(['?'] * len(keys))})"
            cur.execute(stmt)
        self._conn.commit()
        cur.close()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        cur = self._conn.cursor()
        start = 0
        while True:
            query = f"SELECT doc_id FROM {self.table_name}"
            if prefix is not None:
                query += f" AND doc_id LIKE '{prefix}%'"
            query += f" LIMIT {start}, {self.iterator_window_size}"
            cur.execute(query)
            items = cur.fetchall()

            if len(items) == 0:
                break
            for item in items:
                yield item[0]
            start += self.iterator_window_size

        cur.close()


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
    schema.add_field('source', DataType.JSON, max_length=65535)
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
        drop_old=False
    )

    return doc_store
