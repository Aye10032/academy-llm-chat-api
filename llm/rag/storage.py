import sqlite3
from json import JSONDecodeError
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.load import dumps, loads
from langchain_core.stores import BaseStore
from langchain_milvus import Milvus
from langchain_milvus.vectorstores import milvus
from loguru import logger
from pymilvus import DataType, MilvusClient
from pymilvus.orm.types import infer_dtype_bydata
from sqlalchemy import Engine, MetaData, create_engine, Column
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import Field, Session, SQLModel, col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings
from app.db.session import doc_engin, doc_metadata


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
            doc1 = Document(page_content='Hello', metadata={'source': 'doc1'})
            doc2 = Document(page_content='World', metadata={'source': 'doc2'})

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
        iterator_window_size: int = 500,
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
            logger.info(f'创建全文存储数据表 {self.table_name}')

        cur.close()

    def __delete_table(self):
        cur = self._conn.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is not None:
            stmt = f'DROP table {self.table_name}'
            cur.execute(stmt)
            self._conn.commit()
            logger.info(f'删除了全文存储数据表 {self.table_name}')

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

        cur.executemany(f'INSERT INTO {self.table_name} VALUES(?, ?)', data)
        self._conn.commit()
        cur.close()

    def mdelete(self, keys: Sequence[str]) -> None:
        cur = self._conn.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{self.table_name}'")
        if res.fetchone() is None:
            raise ValueError('Collection not found')
        if keys is not None:
            stmt = f'DELETE FROM {self.table_name} WHERE doc_id IN ({",".join(["?"] * len(keys))})'
            cur.execute(stmt)
        self._conn.commit()
        cur.close()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        cur = self._conn.cursor()
        start = 0
        while True:
            query = f'SELECT doc_id FROM {self.table_name}'
            if prefix is not None:
                query += f" AND doc_id LIKE '{prefix}%'"
            query += f' LIMIT {start}, {self.iterator_window_size}'
            cur.execute(query)
            items = cur.fetchall()

            if len(items) == 0:
                break
            for item in items:
                yield item[0]
            start += self.iterator_window_size

        cur.close()


def create_document_model(table_name: str, self_metadata: MetaData):
    class DocumentModel(SQLModel, table=True):
        __tablename__ = table_name
        metadata = self_metadata

        doc_id: str = Field(primary_key=True)
        content: dict = Field(sa_column=Column(JSON))

    return DocumentModel


class SQLStore(BaseStore[str, Document]):
    def __init__(
        self,
        *,
        table_name: str,
        db_url: Optional[Union[str, Path]] = None,
        engine: Optional[Union[Engine, AsyncEngine]] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
        metadata: Optional[MetaData] = None,
        async_mode: Optional[bool] = None,
    ):
        if db_url is None and engine is None:
            raise ValueError('Must specify either db_url or engine')

        if db_url is not None and engine is not None:
            raise ValueError('Must specify either db_url or engine, not both')

        _engine: Union[Engine, AsyncEngine]
        if db_url:
            if async_mode is None:
                async_mode = False
            if async_mode:
                _engine = create_async_engine(
                    url=str(db_url),
                    **(engine_kwargs or {}),
                )
            else:
                _engine = create_engine(url=str(db_url), **(engine_kwargs or {}))
        elif engine:
            _engine = engine
        else:
            raise AssertionError('Something went wrong with configuration of engine.')

        self.engine = _engine
        self.metadata = metadata
        self.table_name = table_name

        self.document_model = create_document_model(table_name, metadata)

    def create_schema(self) -> None:
        if self.metadata:
            self.metadata.create_all(self.engine)
        else:
            SQLModel.metadata.create_all(self.engine)

    async def acreate_schema(self) -> None:
        assert isinstance(self.engine, AsyncEngine)

        async with self.engine.begin() as session:
            if self.metadata:
                await session.run_sync(self.metadata.create_all)
            else:
                await session.run_sync(SQLModel.metadata.create_all)

    def drop(self) -> None:
        self.document_model.__table__.drop(self.engine, checkfirst=True)

    async def amget(self, keys: Sequence[str]) -> list[Optional[Document]]:
        async with AsyncSession(self.engine) as session:
            query = select(self.document_model).where(col(self.document_model.doc_id).in_(keys))
            result = await session.exec(query)
            docs = result.all()

            ordered_values = {key: type[Document] for key in keys}
            for doc in docs:
                val = Document.model_validate(doc.content)
                val.metadata['doc_id'] = doc.doc_id
                ordered_values[doc.doc_id] = val

            return [ordered_values[key] for key in keys]

    def mget(self, keys: Sequence[str]) -> list[Optional[Document]]:
        with Session(self.engine) as session:
            query = select(self.document_model).where(col(self.document_model.doc_id).in_(keys))
            result = session.exec(query)
            docs = result.all()

            ordered_values = {key: type[Document] for key in keys}
            for doc in docs:
                val = Document.model_validate(doc.content)
                val.metadata['doc_id'] = doc.doc_id
                ordered_values[doc.doc_id] = val

            return [ordered_values[key] for key in keys]

    async def amset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        with AsyncSession(self.engine) as session:
            for doc_id, doc in key_value_pairs:
                content = doc.model_dump()
                db_doc = self.document_model(doc_id=doc_id, content=content)
                session.add(db_doc)
            await session.commit()

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        with Session(self.engine) as session:
            for doc_id, doc in key_value_pairs:
                content = doc.model_dump()
                db_doc = self.document_model(doc_id=doc_id, content=content)
                session.add(db_doc)
            session.commit()

    async def amdelete(self, keys: Sequence[str]) -> None:
        async with AsyncSession(self.engine) as session:
            query = select(self.document_model).where(col(self.document_model.doc_id).in_(keys))
            results = session.exec(query)
            docs = results.all()
            await session.delete(docs)
            await session.commit()

    def mdelete(self, keys: Sequence[str]) -> None:
        with Session(self.engine) as session:
            query = select(self.document_model).where(col(self.document_model.doc_id).in_(keys))
            results = session.exec(query)
            docs = results.all()
            session.delete(docs)
            session.commit()

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:
        async with AsyncSession(self.engine) as session:
            query = select(self.document_model.doc_id)
            if prefix:
                query = query.where(col(self.document_model.doc_id).like(f'{prefix}%'))
            results = await session.exec(query)
            for (doc_id,) in results:
                yield doc_id

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        with Session(self.engine) as session:
            query = select(self.document_model.doc_id)
            if prefix:
                query = query.where(col(self.document_model.doc_id).like(f'{prefix}%'))
            results = session.exec(query)
            for (doc_id,) in results:
                yield doc_id


def create_vector_db(
    table_name: str,
    embedding_model: Embeddings,
    *,
    db_name: str = 'default',
    index_params: Optional[dict[str, Any]] = None,
    drop_old: bool = False,
) -> Milvus:
    """初始化Milvus数据库

    Args:
        table_name: 数据表名称
        embedding_model: 向量查询所使用的嵌入模型
        db_name: 连接的数据库，默认为 'default'.
        index_params: （可选参数）自定义索引算法参数
        drop_old: 是否覆盖旧数据库

    Returns:
        Milvus: langchain格式的Milvus数据库对象
    """
    vector_field_embeddings = embedding_model.embed_documents(['test'])
    dim = len(vector_field_embeddings[0])

    client = MilvusClient(**get_settings().knowledge_base.milvus.get_conn_args())
    if db_name not in client.list_databases():
        client.create_database(db_name)
    client.use_database(db_name)

    if drop_old and client.has_collection(table_name):
        client.drop_collection(table_name)
        logger.info(f'删除了向量数据表 {table_name}')

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
    schema.add_field('file_id', DataType.VARCHAR, max_length=65535)
    schema.add_field('has_full_text', DataType.BOOL, default_value=True)
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
        index_params=index_params,
    )
    client.close()

    vector_db = get_vector_db(table_name, embedding_model, db_name=db_name)

    return vector_db


def get_vector_db(
    table_name: str, embedding_model: Embeddings, *, db_name: str = 'default', ef: int = 10
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

    milvus_cfg = get_settings().knowledge_base.milvus
    vector_db: milvus = Milvus(
        embedding_model,
        collection_name=table_name,
        connection_args=milvus_cfg.get_conn_args(db_name),
        search_params={'metric_type': 'L2', 'params': {'ef': ef}},
        auto_id=True,
    )

    return vector_db


def get_doc_db(table_name: str, *, drop_old: bool = False) -> BaseStore:
    # doc_store = SqliteDocStore(
    #     connection_string=get_settings().knowledge_base.DOC_URL,
    #     table_name=table_name,
    #     drop_old=drop_old,
    # )
    doc_store = SQLStore(table_name=table_name, engine=doc_engin, metadata=doc_metadata)
    if drop_old:
        doc_store.drop()
        doc_store.create_schema()

    return doc_store


def fix_null_fields(docs: list[Document]) -> list[Document]:
    """临时修复zilliz文档处理

    目前，zilliz无法正确处理空字端，手动补全剩余字段
    """
    required_fields = [
        'section',
        'section_3',
        'section_4',
        'section_5',
        'section_6',
    ]
    for doc in docs:
        for field in required_fields:
            doc.metadata.setdefault(field, '')

    return docs
