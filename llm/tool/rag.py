from typing import Optional, Type

from langchain.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from sqlmodel import Session

import app.crud.knowledge_base as kb_crud
from app.db.session import engine
from llm.core.model import load_embedding, load_llm
from llm.core.template import SELECT_KNOWLEDGE_BASE_SYSTEM_ZH
from llm.rag.retriever import ExprRetriever
from llm.rag.storage import get_doc_db, get_vector_db


class SelectKnowledgeBaseInput(BaseModel):
    query: str = Field(description='用户的原始提问文本')


class SelectKnowledgeBaseOutput(BaseModel):
    """合适的提问问句与所查询知识库
    此外，根据用户提问的内容，决定这个问题是否适合先进行文章级别的搜索，再进行文本内容的查找。
    """

    question: str = Field(
        description='根据所需要的信息分析的来的，具体用于从知识库中找回文本的语句'
    )
    table_name: str = Field(description='查询的知识库名称。如果没有合适的知识库，则留空。')
    paper_first: bool = Field(description='用户的搜索请求是否适合先进行文章检索再进行内容检索？')


class SelectKnowledgeBase(BaseTool):
    name: str = 'select_vecstore'
    description: str = '向量知识库查询工具，能够根据用户的需求自行判断最合适的数据库进行查询'
    args_schema: Type[BaseModel] = SelectKnowledgeBaseInput
    return_direct: bool = False
    handle_tool_error: bool = True

    llm: Optional[ChatOpenAI] = None
    available_knowledge_bases: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def init_llm(self):
        if self.llm is None:
            self.llm = load_llm('gpt-4o')

        return self

    def _run(
        self, query: str, config: Optional[RunnableConfig] = None
    ) -> SelectKnowledgeBaseOutput:
        with Session(engine) as session:
            kb_list = kb_crud.get_list(session, 0, 20)

        if self.available_knowledge_bases:
            available_kbs = '\n=================\n'.join(
                [
                    f'name: {kb.table_name}\ndescription: {kb.description}'
                    for kb in kb_list
                    if kb.uid in self.available_knowledge_bases
                ]
            )
        else:
            available_kbs = '\n=================\n'.join(
                [f'name: {kb.table_name}\ndescription: {kb.description}' for kb in kb_list]
            )

        llm = self.llm.with_structured_output(SelectKnowledgeBaseOutput, include_raw=True)
        prompt = ChatPromptTemplate.from_messages(
            [('system', SELECT_KNOWLEDGE_BASE_SYSTEM_ZH), ('human', '{human_input}')]
        )
        chain = prompt | llm
        result: SelectKnowledgeBaseOutput = chain.invoke(
            {'available_kbs': available_kbs, 'human_input': query}, config
        )

        return result


class RAGSearchInput(BaseModel):
    question: str = Field(description='用于从知识库中召回长文本的搜索文本')
    table_name: str = Field(description='具体使用的知识库名称')
    expr: str = Field(description='向量数据库查询条件语句')


class RAGSearchTool(BaseTool):
    name: str = 'search_from_vecstore'
    description: str = '根据查询文本从指定向量数据库中搜索相关的知识。如果用户已经给出了使用哪个数据库，可以直接传入相应的参数调用此工具进行查询'
    args_schema: Type[BaseModel] = RAGSearchInput
    return_direct: bool = False
    handle_tool_error: bool = True

    def _run(
        self,
        question: str,
        table_name: str,
        expr: str,
        config: Optional[RunnableConfig] = None,
    ) -> list[Document]:
        """从向量数据库中进行查询操作"""
        logger.info(f'Calling VecstoreSearchTool with query {question}')

        embedding = load_embedding()

        vec_store = get_vector_db(table_name, embedding, db_name='llm_chat')
        doc_store = get_doc_db(table_name)

        if not expr:
            retriever = MultiVectorRetriever(
                vectorstore=vec_store,
                docstore=doc_store,
                search_kwargs={'k': 8, 'fetch_k': 10},
            )

        else:
            retriever = ExprRetriever(
                vectorstore=vec_store,
                docstore=doc_store,
                search_kwargs={'k': 8, 'fetch_k': 10},
                expr_statement=expr,
            )

        output = retriever.invoke(question, config)
        return output
