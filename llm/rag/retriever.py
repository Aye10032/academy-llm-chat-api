from typing import Optional, Any, Literal

from langchain.retrievers import ParentDocumentRetriever, MultiVectorRetriever
from langchain.retrievers.multi_query import LineListOutputParser
from langchain.retrievers.multi_vector import SearchType
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus
from langchain_milvus.vectorstores import milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pymilvus import DataType, MilvusClient
from pymilvus.orm.types import infer_dtype_bydata

from app.core.config import get_settings
from llm.core.embedding_core import BgeReranker
from llm.core.model_core import load_gpt4o_mini
from llm.core.template import MULTIQUERY_SYSTEM_EN, MULTIQUERY_HUMAN_EN
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
        drop_old=False
    )

    return doc_store


def _unique_doc(docs: list[Document]) -> list[Document]:
    result = []
    for doc in docs:
        if doc not in result:
            result.append(doc)

    return result


def _get_parent_id(docs: list[Document], id_key: str) -> tuple[list, dict[str, Any]]:
    ids = []
    id_map = {}
    for sentence in docs:
        if id_key in sentence.metadata:
            doc_id = sentence.metadata[id_key]
            if doc_id not in ids:
                ids.append(doc_id)
                id_map[doc_id] = [sentence.page_content]
            else:
                temp: list = id_map.get(doc_id)
                temp.append(sentence.page_content)
                id_map[doc_id] = temp

    return ids, id_map


class ScoreRetriever(MultiVectorRetriever):
    reranker: BgeReranker

    multi_query: bool = False
    llm_chain: Optional[Runnable] = None

    top_k: int = 5

    def generate_queries(
            self, question: str, run_manager: CallbackManagerForRetrieverRun
    ) -> list[str]:
        response = self.llm_chain.invoke(
            {'question': question}, config={'callbacks': run_manager.get_child()}
        )

        return response

    def retrieve_documents(
            self, queries: list[str], run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        documents = [
            self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
            if self.search_type == SearchType.mmr
            else self.vectorstore.similarity_search(query, **self.search_kwargs)
            for query in queries
        ]
        return documents

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.multi_query:
            queries = self.generate_queries(query, run_manager)
            queries.append(query)
            short_doc = self.retrieve_documents(queries, run_manager)
            short_doc = _unique_doc(short_doc)
        else:
            if self.search_type == SearchType.similarity:
                short_doc = self.vectorstore.similarity_search(query, **self.search_kwargs)
            else:
                short_doc = self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)

        ids, id_map = _get_parent_id(short_doc, self.id_key)

        docs = self.docstore.mget(ids)
        logger.info(f'retrieve {len(docs)} documents, reranking...')

        try:
            rerank_docs = self.reranker.compress_documents(docs, query)[:self.top_k]

            for i in range(len(rerank_docs)):
                context_id = rerank_docs[i].metadata[self.id_key]
                rerank_docs[i].metadata['refer_sentence'] = id_map.get(context_id) if context_id in id_map else []

            return rerank_docs
        except Exception as e:
            logger.error(f'catch exception {e} while check {ids}')

    async def agenerate_queries(
            self, question: str, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[str]:
        response = await self.llm_chain.ainvoke(
            {'question': question}, callbacks=run_manager.get_child()
        )

        return response


def insert_chain(
        vector_store: VectorStore,
        doc_store: BaseStore,
        language: str = Literal['en', 'zh']
) -> ParentDocumentRetriever:
    """用于插入新文档的任务链

    Args:
        vector_store: 向量数据库
        doc_store: 存储长分片的KV数据库
        language: 输入文档的语言

    Returns:
        ParentDocumentRetriever
    """
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


def base_retriever(
        vector_store: VectorStore,
        doc_store: BaseStore,
        reranker: BgeReranker
) -> ScoreRetriever:
    # TODO 视情况决定是否翻译句子
    retriever_llm = load_gpt4o_mini()
    query_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=MULTIQUERY_SYSTEM_EN),
        ('human', MULTIQUERY_HUMAN_EN)
    ])

    parser = LineListOutputParser()

    llm_chain = query_prompt | retriever_llm | parser

    retriever = ScoreRetriever(
        vectorstore=vector_store,
        docstore=doc_store,
        reranker=reranker,
        multi_query=True,
        llm_chain=llm_chain,
        search_type=SearchType.similarity,
        search_kwargs={'k': 8, 'fetch_k': 10},
        top_k=5
    )

    return retriever
