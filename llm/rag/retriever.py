from typing import Any, Literal, Optional

from langchain.retrievers import MultiVectorRetriever, ParentDocumentRetriever
from langchain.retrievers.multi_query import LineListOutputParser
from langchain.retrievers.multi_vector import SearchType
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from llm.core.model import BgeReranker, load_llm
from llm.core.template import MULTIQUERY_HUMAN_EN, MULTIQUERY_SYSTEM_EN


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
        documents = []
        search_func = (
            self.vectorstore.similarity_search
            if self.search_type == SearchType.similarity
            else self.vectorstore.max_marginal_relevance_search
        )
        logger.debug(queries)
        for query in queries:
            short_doc = search_func(query=query, **self.search_kwargs)
            documents.extend(short_doc)
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
                short_doc = self.vectorstore.max_marginal_relevance_search(
                    query, **self.search_kwargs
                )

        ids, id_map = _get_parent_id(short_doc, self.id_key)

        docs = self.docstore.mget(ids)
        logger.info(f'retrieve {len(docs)} documents, reranking...')

        try:
            rerank_docs = self.reranker.compress_documents(docs, query)[: self.top_k]

            for i in range(len(rerank_docs)):
                context_id = rerank_docs[i].metadata[self.id_key]
                rerank_docs[i].metadata['refer_sentence'] = (
                    id_map.get(context_id) if context_id in id_map else []
                )

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


class ExprRetriever(MultiVectorRetriever):
    expr_statement: str

    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.search_type == SearchType.similarity:
            short_doc: list[Document] = self.vectorstore.similarity_search(
                query, expr=self.expr_statement, **self.search_kwargs
            )
        else:
            short_doc: list[Document] = self.vectorstore.max_marginal_relevance_search(
                query, expr=self.expr_statement, **self.search_kwargs
            )

        ids, id_map = _get_parent_id(short_doc, self.id_key)

        docs = self.docstore.mget(ids)
        for i in range(len(docs)):
            context_id = docs[i].metadata[self.id_key]
            docs[i].metadata['refer_sentence'] = id_map.get(context_id)

        return docs


def insert_chain(
    vector_store: VectorStore, doc_store: BaseStore, language: str = Literal['en', 'zh']
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
        separators=['\n\n', '\t\n'], keep_separator=False
    )

    if language == 'zh':
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=0, separators=['。', '. '], keep_separator=False
        )
    else:
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=0, separators=['. '], keep_separator=False
        )

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever


def base_retriever(
    vector_store: VectorStore, doc_store: BaseStore, reranker: BgeReranker
) -> ScoreRetriever:
    # TODO 视情况决定是否翻译句子
    retriever_llm = load_llm('gpt-4o-mini')
    query_prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=MULTIQUERY_SYSTEM_EN), ('human', MULTIQUERY_HUMAN_EN)]
    )

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
        top_k=5,
    )

    return retriever


def format_docs(docs: list[Document]) -> str:
    formatted = [
        (
            f'<doc-id>{i + 1}</doc-id>\n'
            f'<doc-title>{doc.metadata["title"]}</doc-title>\n'
            f'<doc-author>{doc.metadata["author"]}</doc-author>\n'
            f'<doc-year>{doc.metadata["year"]}</doc-year>\n'
            f'<doc-content>{doc.page_content}</doc-content>'
        )
        for i, doc in enumerate(docs)
    ]

    return '\n\n'.join(formatted)
