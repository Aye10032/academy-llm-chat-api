import operator

from typing import Annotated, Literal

from langchain_core.messages import SystemMessage, ToolMessage, ToolCall
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.constants import START
from langgraph.types import Command
from langgraph.graph import StateGraph, END, MessagesState
from tqdm import tqdm

from llm.core.model import load_deepseek_v3, load_reranker
from llm.core.template import CONCLUDE_DOCUMENTS_SYSTEM_ZH
from llm.file_loader.web import JinaWebLoader
from llm.rag.retriever import format_docs
from llm.tool.rag import RAGSearchTool, SelectKnowledgeBase, SelectKnowledgeBaseOutput
from llm.tool.search import WebSearchTool


class SearchAgentState(MessagesState):
    origin_question: str
    search_results: Annotated[list[Document], operator.add]
    sources: Annotated[list[str], operator.add]


class SearchAgent:
    select_tool = SelectKnowledgeBase()
    rag_search_tool = RAGSearchTool()
    web_search_tool = WebSearchTool()

    llm = load_deepseek_v3()

    def __init__(self, use_web=True):
        self.use_web = use_web
        if use_web:
            self.tools = [
                self.select_tool,
                self.rag_search_tool,
                self.web_search_tool
            ]
        else:
            self.tools = [
                self.select_tool,
                self.rag_search_tool
            ]

    def search_agent(self, state: SearchAgentState):
        llm_with_tool = self.llm.bind_tools(self.tools, tool_choice='required')
        messages = state['messages']
        response = llm_with_tool.invoke(messages)
        return {'messages': [response]}

    def choose_tool(self, state: SearchAgentState):
        messages = state['messages']
        tool_calls = messages[-1].tool_calls

        tool_call = tool_calls[0]
        if tool_call['name'] == self.web_search_tool.name:
            return 'web_tool'
        else:
            return 'rag_tool'

    def rag_tool_node(self, state: SearchAgentState) -> Command[Literal['search_agent', 'search_conclude']]:
        messages = state['messages']
        tool_call: ToolCall = messages[-1].tool_calls[0]
        tool_call_id = tool_call['id']

        if tool_call['name'] == self.select_tool.name:
            select_result: SelectKnowledgeBaseOutput = self.select_tool.invoke(tool_call['args'])

            if not select_result.table_name:
                if self.use_web:
                    return Command(
                        update={
                            'messages': [
                                ToolMessage('没有合适的向量数据库，向量数据库查询失败。', tool_call_id=tool_call_id)
                            ]
                        }, goto='search_agent'
                    )
                else:
                    return Command(
                        update={
                            'search_results': []
                        },
                        goto='search_conclude'
                    )

            search_result = self.rag_search_tool.invoke({
                'query': select_result.question,
                'target_collection': select_result.table_name
            })
        else:
            search_result = self.rag_search_tool.invoke(tool_call['args'])

        if search_result or not self.use_web:
            return Command(
                update={
                    'search_results': search_result
                },
                goto='search_conclude'
            )

        return Command(
            update={
                'messages': [
                    ToolMessage('向量数据库查询失败', tool_call_id=tool_call_id)
                ]
            },
            goto='search_agent'
        )

    def web_tool_node(self, state: SearchAgentState):
        messages = state['messages']
        tool_call: ToolCall = messages[-1].tool_calls[0]

        search_urls = self.web_search_tool.invoke(tool_call['args'])

        all_web_docs = []
        for url in tqdm(search_urls, total=len(search_urls)):
            web_loader = JinaWebLoader()
            _, docs = web_loader.load(url)
            all_web_docs.extend(docs)

        return {'search_results': all_web_docs}

    def search_conclude_node(self, state: SearchAgentState):
        reranker = load_reranker()
        clean_output = reranker.compress_documents(state['search_results'], state['origin_question'])

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(CONCLUDE_DOCUMENTS_SYSTEM_ZH),
            ('human', '待总结的文档片段：\n\n{doc_str}')
        ])
        chain = prompt | self.llm
        response = chain.invoke({
            'doc_str': format_docs(clean_output, 'zh')
        })

        return {'messages': [response]}

    def build(self):
        searcher = StateGraph(SearchAgentState)
        searcher.add_node('search_agent', self.search_agent)
        searcher.add_node('rag_tool', self.rag_tool_node)
        searcher.add_node('web_tool', self.web_tool_node)
        searcher.add_node('search_conclude', self.search_conclude_node)

        searcher.add_edge(START, 'search_agent')
        searcher.add_conditional_edges('search_agent', self.choose_tool)
        searcher.add_edge('web_tool', 'search_conclude')
        searcher.add_edge('search_conclude', END)

        return searcher.compile()
