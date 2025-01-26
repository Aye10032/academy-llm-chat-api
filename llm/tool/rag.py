from typing import Type, Optional

from langchain.retrievers import MultiVectorRetriever
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, ToolException
from loguru import logger
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.crud.knowledge_base import get_knowledge_bases
from app.db.session import engine
from llm.core.model import load_embedding, load_gpt4o, load_deepseek_v3, load_reranker
from llm.core.template import SELECT_KNOWLEDGE_BASE_SYSTEM_ZH
from llm.rag.storage import get_vector_db, get_doc_db


class SelectKnowledgeBaseInput(BaseModel):
    query: str = Field(description='用户的原始提问文本')


class SelectKnowledgeBaseOutput(BaseModel):
    """合适的提问问句与所查询知识库"""
    question: str = Field(description='根据所需要的信息分析的来的，具体用于从知识库中找回文本的语句')
    table_name: str = Field(description='查询的知识库名称。如果没有合适的知识库，则留空。')


class SelectKnowledgeBase(BaseTool):
    name: str = 'select_vecstore'
    description: str = '如果用户没有指定具体使用哪个知识库，则调用此工具分析问题并得到合适的数据库'
    args_schema: Type[BaseModel] = SelectKnowledgeBaseInput
    return_direct: bool = False
    handle_tool_error: bool = True

    def _run(self, query: str) -> SelectKnowledgeBaseOutput:
        with Session(engine) as session:
            kb_list = get_knowledge_bases(session, 0, 20)

        available_kbs = '\n=================\n'.join([
            f'name: {kb.table_name}\ndescription: {kb.description}'
            for kb in kb_list
        ])

        llm = load_deepseek_v3().with_structured_output(SelectKnowledgeBaseOutput)
        prompt = ChatPromptTemplate.from_messages([
            ('system', SELECT_KNOWLEDGE_BASE_SYSTEM_ZH),
            ('human', '{human_input}')
        ])
        chain = prompt | llm
        result: SelectKnowledgeBaseOutput = chain.invoke({
            'available_kbs': available_kbs,
            'human_input': query
        })

        return result


class RAGSearchInput(BaseModel):
    query: str = Field(description='用于从知识库中召回长文本的搜索文本')
    target_collection: str = Field(description='具体使用的知识库名称')


class RAGSearchTool(BaseTool):
    name: str = 'search_from_vecstore'
    description: str = '根据查询文本从指定向量数据库中搜索相关的知识。如果用户已经给出了使用哪个数据库，可以直接传入相应的参数调用此工具进行查询'
    args_schema: Type[BaseModel] = RAGSearchInput
    return_direct: bool = False
    handle_tool_error: bool = True

    def _run(
            self,
            query: str,
            target_collection: str,
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> list[Document]:
        """从向量数据库中进行查询操作"""
        logger.info(f'Calling VecstoreSearchTool with query {query}')

        embedding = load_embedding()

        vec_store = get_vector_db(target_collection, embedding, db_name='llm_chat')
        doc_store = get_doc_db(target_collection)

        retriever = MultiVectorRetriever(
            vectorstore=vec_store,
            docstore=doc_store,
            search_kwargs={'k': 8, 'fetch_k': 10}
        )

        output = retriever.invoke(query)
        return output
