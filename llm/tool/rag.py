from typing import Type, Optional

from langchain.retrievers import MultiVectorRetriever
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

from llm.core.model import load_embedding
from llm.rag.storage import get_vector_db, get_doc_db


class RAGSearchInput(BaseModel):
    query: str = Field(description='用于从知识库中召回长文本的搜索文本')


class RAGSearchTool(BaseTool):
    name: str = 'search_from_vecstore'
    description: str = '根据查询文本从向量数据库中搜索相关的知识。AI在写作过程中，如遇到不明确的知识点或术语，可以调用此工具从数据库中进行查询以获取相关信息。'
    args_schema: Type[BaseModel] = RAGSearchInput
    return_direct: bool = False
    handle_tool_error: bool = True

    target_collection: Optional[str] = None

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> list[Document]:
        """从向量数据库中进行查询操作"""
        logger.info(f'Calling VecstoreSearchTool with query {query}')

        embedding = load_embedding()

        vec_store = get_vector_db(self.target_collection, embedding, db_name='llm_chat')
        doc_store = get_doc_db(self.target_collection)

        retriever = MultiVectorRetriever(
            vectorstore=vec_store,
            docstore=doc_store,
            search_kwargs={'k': 8, 'fetch_k': 10}
        )

        output = retriever.invoke(query)

        return output
