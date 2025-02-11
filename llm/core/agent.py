import operator

from typing import Annotated, Literal, Optional
from uuid import uuid4

from langchain_core.messages import SystemMessage, ToolMessage, ToolCall, AIMessage, AnyMessage, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langgraph.graph import StateGraph, END, MessagesState, add_messages
from pydantic import BaseModel, model_validator, Field
from sqlmodel import Session
from tqdm import tqdm

from app.crud.manuscript import insert_manuscript, get_drafts
from app.db.session import engine
from app.models import ManuscriptTable
from app.schemas.manuscript import Manuscript
from llm.core.model import load_reranker, load_gpt4o, load_gpt4o_mini
from llm.core.template import CONCLUDE_DOCUMENTS_SYSTEM_ZH, OPTIMIZER_SYSTEM_ZH, CONCLUDE_DOCUMENTS_HUMAN_ZH, AGENT_SYSTEM_ZH, \
    GENERATOR_ROUTE_SYSTEM_ZH, GENERATOR_SYSTEM_ZH, GENERATOR_HUMAN_ZH, KNOWLEDGE_MANAGE_SYSTEM_ZH
from llm.file_loader.web import JinaWebLoader
from llm.rag.retriever import format_docs
from llm.schemas import MarkdownMeta
from llm.schemas.tokens import UsageMetadata
from llm.tool.modify import Modifier, OptimizerOutput, Rewriter, RewriterOutput
from llm.tool.rag import RAGSearchTool, SelectKnowledgeBase, SelectKnowledgeBaseOutput
from llm.tool.search import WebSearchTool


class BaseAgentState(MessagesState):
    project_uid: str
    price: Annotated[float, operator.add]


class KnowledgeManageAgentState(BaseAgentState):
    documents: Annotated[list[Document], operator.add]


class KnowledgeManageAgent(BaseModel):
    use_web: bool = True
    available_knowledge_bases: list[str] = Field(default_factory=list)
    router_llm: ChatOpenAI
    task_llm: ChatOpenAI

    tools: list[BaseTool] = Field(default_factory=list)
    select_tool: Optional[SelectKnowledgeBase] = None
    rag_search_tool: Optional[RAGSearchTool] = None
    web_search_tool: Optional[WebSearchTool] = None

    @model_validator(mode='after')
    def setup_tools(self):
        self.select_tool = SelectKnowledgeBase(llm=self.router_llm, available_knowledge_bases=self.available_knowledge_bases)
        self.rag_search_tool = RAGSearchTool()
        self.web_search_tool = WebSearchTool()

        if self.use_web:
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

        return self

    def search_router(self, state: KnowledgeManageAgentState):
        llm_with_tool = self.router_llm.bind_tools(self.tools)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=KNOWLEDGE_MANAGE_SYSTEM_ZH),
            MessagesPlaceholder(variable_name='history')
        ])

        chain = prompt | llm_with_tool
        response: AIMessage = chain.invoke({'history': state['messages']})

        token_usage = UsageMetadata.create(response.usage_metadata)
        return {
            'messages': [response],
            'price': token_usage.calculate_cost(self.router_llm)
        }

    def choose_tool(self, state: KnowledgeManageAgentState) -> Literal['web_tool', 'rag_select', 'rag_search', '__end__']:
        messages = state['messages']
        tool_calls = messages[-1].tool_calls

        if tool_calls:
            tool_call = tool_calls[0]
            if tool_call['name'] == self.web_search_tool.name:
                return 'web_tool'
            elif tool_call['name'] == self.select_tool.name:
                return 'rag_select'
            elif tool_call['name'] == self.rag_search_tool.name:
                return 'rag_search'
        else:
            return '__end__'

    def rag_select_node(self, state: KnowledgeManageAgentState) -> Command[Literal['search_router', 'rag_search']]:
        messages = state['messages']
        tool_call: ToolCall = messages[-1].tool_calls[0]
        tool_call_id = tool_call['id']

        select_dict = self.select_tool.invoke(tool_call['args'])

        select_result: SelectKnowledgeBaseOutput = select_dict['parsed']
        token_usage = UsageMetadata.create(select_dict['raw'].usage_metadata)

        if not select_result.table_name:
            if self.use_web:
                return Command(
                    update={
                        'messages': [
                            ToolMessage('没有合适的向量数据库，向量数据库查询失败，请尝试联网查询。', tool_call_id=tool_call_id)
                        ],
                        'price': token_usage.calculate_cost(self.router_llm)
                    }, goto='search_router'
                )
            else:
                return Command(
                    update={
                        'messages': [
                            ToolMessage(
                                '没有合适的向量数据库，向量数据库查询失败，请将这个结果反馈给用户，并询问用户是否考虑联网搜索。',
                                tool_call_id=tool_call_id
                            )
                        ],
                        'price': token_usage.calculate_cost(self.router_llm)
                    },
                    goto='search_router'
                )

        messages.pop(-1)
        return Command(
            update={
                'messages': [
                    AIMessage(
                        '', tool_calls=[
                            ToolCall(
                                name=self.rag_search_tool.name,
                                args=select_result.model_dump(),
                                id=tool_call_id
                            )
                        ]
                    )
                ],
                'price': token_usage.calculate_cost(self.router_llm)
            },
            goto='rag_search'
        )

    def rag_search_node(self, state: KnowledgeManageAgentState) -> Command[Literal['search_router', 'search_conclude']]:
        messages = state['messages']
        tool_call: ToolCall = messages[-1].tool_calls[0]
        tool_call_id = tool_call['id']

        search_result = self.rag_search_tool.invoke(tool_call['args'])

        if search_result:
            return Command(
                update={
                    'documents': search_result,
                },
                goto='search_conclude'
            )

        if self.use_web:
            return Command(
                update={
                    'messages': [
                        ToolMessage('向量数据库查询失败，请尝试联网查询。', tool_call_id=tool_call_id)
                    ]
                },
                goto='search_router'
            )
        else:
            return Command(
                update={
                    'messages': [
                        ToolMessage(
                            '没有合适的向量数据库，向量数据库查询失败，请将这个结果反馈给用户，并询问用户是否考虑联网搜索。',
                            tool_call_id=tool_call_id
                        )
                    ],
                },
                goto='search_router'
            )

    def web_tool_node(self, state: KnowledgeManageAgentState):
        messages = state['messages']
        tool_call: ToolCall = messages[-1].tool_calls[0]

        search_urls = self.web_search_tool.invoke(tool_call['args'])

        all_web_docs = []
        for url in tqdm(search_urls, total=len(search_urls)):
            web_loader = JinaWebLoader()
            _, docs = web_loader.load(url)
            all_web_docs.extend(docs)

        return {'documents': all_web_docs}

    def search_conclude_node(self, state: KnowledgeManageAgentState):
        origin_question = state['messages'][-1].content
        reranker = load_reranker()
        clean_output = reranker.compress_documents(state['documents'], origin_question)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(CONCLUDE_DOCUMENTS_SYSTEM_ZH),
            ('human', CONCLUDE_DOCUMENTS_HUMAN_ZH)
        ])
        chain = prompt | self.task_llm
        response = chain.invoke({
            'question': origin_question,
            'doc_str': format_docs(clean_output, 'zh')
        })

        lines = response.content.splitlines()
        title = lines[0]

        manuscript = ManuscriptTable(
            uid=str(uuid4()),
            project_uid=state['project_uid'],
            title=title.lstrip('#').strip(),
            content=response.content,
            is_draft=True
        )
        with Session(engine) as session:
            insert_manuscript(session, manuscript)

        token_usage = UsageMetadata.create(response.usage_metadata)
        return {
            'messages': [AIMessage(content=f'我已经找到了相关的资料，并将总结的资料保存到了文件《{title.strip("#").strip()}》中。')],
            'price': token_usage.calculate_cost(self.router_llm)
        }

    def build(self) -> CompiledStateGraph:
        searcher = StateGraph(KnowledgeManageAgentState)
        searcher.add_node('search_router', self.search_router)
        searcher.add_node('rag_select', self.rag_select_node)
        searcher.add_node('rag_search', self.rag_search_node)
        searcher.add_node('web_tool', self.web_tool_node)
        searcher.add_node('search_conclude', self.search_conclude_node)

        searcher.add_edge(START, 'search_router')
        searcher.add_conditional_edges('search_router', self.choose_tool)
        searcher.add_edge('web_tool', 'search_conclude')
        searcher.add_edge('search_conclude', END)

        return searcher.compile()


class OptimizerAgentState(BaseAgentState):
    current_text: str


class OptimizerAgent(BaseModel):
    router_llm: ChatOpenAI
    task_llm: ChatOpenAI

    modifier: Optional[Modifier] = None
    rewriter: Optional[Rewriter] = None

    @model_validator(mode='after')
    def setup_tools(self):
        self.modifier = Modifier(llm=self.task_llm)
        self.rewriter = Rewriter(llm=self.task_llm)

        return self

    def optimizer_route_node(self, state: OptimizerAgentState):
        tools = [self.modifier, self.rewriter]
        llm_with_tool = self.router_llm.bind_tools(tools)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=OPTIMIZER_SYSTEM_ZH),
            MessagesPlaceholder(variable_name='history')
        ])

        chain = prompt | llm_with_tool
        response = chain.invoke({'history': state['messages']})

        token_usage = UsageMetadata.create(response.usage_metadata)
        return {
            'messages': [response],
            'price': token_usage.calculate_cost(self.router_llm)
        }

    def optimizer_route(self, state: OptimizerAgentState) -> Literal['rewriter', 'modifier', '__end__']:
        message: AIMessage = state['messages'][-1]

        if message.tool_calls:
            tool_call = message.tool_calls[0]

            if tool_call['name'] == self.modifier.name:
                return 'modifier'
            elif tool_call['name'] == self.rewriter.name:
                return 'rewriter'

        return '__end__'

    def rewriter_node(self, state: OptimizerAgentState) -> Command[Literal['optimizer_router']]:
        message: AIMessage = state['messages'][-1]
        tool_call = message.tool_calls[0]
        tool_call_id = tool_call['id']

        response = self.rewriter.invoke({
            'query': tool_call['args']['query'],
            'current_text': state['current_text']
        })
        modify_result: RewriterOutput = response['parsed']
        token_usage = UsageMetadata.create(response['raw'].usage_metadata)

        return Command(
            update={
                'messages': [
                    ToolMessage(f'我已经完成了重写：{modify_result.explanation}', tool_call_id=tool_call_id)
                ],
                'current_text': modify_result.rewrite,
                'price': token_usage.calculate_cost(self.router_llm)
            }, goto='optimizer_router'
        )

    def modify_node(self, state: OptimizerAgentState) -> Command[Literal['optimizer_router']]:
        message: AIMessage = state['messages'][-1]
        tool_call = message.tool_calls[0]
        tool_call_id = tool_call['id']

        response = self.modifier.invoke({
            'query': tool_call['args']['query'],
            'current_text': state['current_text']
        })
        modify_result: OptimizerOutput = response['parsed']
        token_usage = UsageMetadata.create(response['raw'].usage_metadata)

        return Command(
            update={
                'messages': [
                    ToolMessage(f'我已经完成了修改：{str(modify_result.model_dump())}', tool_call_id=tool_call_id)
                ],
                'price': token_usage.calculate_cost(self.router_llm)
            }, goto='optimizer_router'
        )

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(OptimizerAgentState)
        graph.add_node('optimizer_router', self.optimizer_route_node)
        graph.add_node('rewriter', self.rewriter_node)
        graph.add_node('modifier', self.modify_node)

        graph.add_edge(START, 'optimizer_router')
        graph.add_conditional_edges('optimizer_router', self.optimizer_route)

        return graph.compile()


class GenerateAgentState(BaseAgentState):
    write_request: str
    current_text: str
    information_list: Annotated[list[Manuscript], operator.add]


class GenerateAgent(BaseModel):
    router_llm: ChatOpenAI
    task_llm: ChatOpenAI

    @staticmethod
    @tool
    def analyzer():
        """本地知识检索工具
        对于需要比较复杂知识的写作任务，可以先调用此工具。他会浏览本地的知识库并分析是否已经有查找好的可用于本次写作任务的资料。
        """
        return

    @staticmethod
    @tool
    def generator():
        """文本生成工具
        此工具能够根据用户的需要生成符合要求的文本"""
        return

    def generator_route_node(self, state: GenerateAgentState):
        last_message = state['messages'][-1].content

        tools = [self.analyzer]
        llm_with_tool = self.router_llm.bind_tools(tools)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=GENERATOR_ROUTE_SYSTEM_ZH),
            MessagesPlaceholder(variable_name='history')
        ])

        chain = prompt | llm_with_tool
        response = chain.invoke({'history': state['messages']})

        token_usage = UsageMetadata.create(response.usage_metadata)

        if 'write_request' not in state:
            return {
                'write_request': last_message,
                'current_text': '',
                'messages': [response],
                'price': token_usage.calculate_cost(self.router_llm)
            }
        else:
            return {
                'messages': [response],
                'price': token_usage.calculate_cost(self.router_llm)
            }

    def generator_route(self, state: GenerateAgentState) -> Literal['analyzer', 'generator', '__end__']:
        messages = state['messages']
        tool_calls = messages[-1].tool_calls

        if tool_calls:
            tool_call = tool_calls[0]
            if tool_call['name'] == self.analyzer.name:
                return 'analyzer'
            elif tool_call['name'] == self.generator.name:
                return 'generator'
        else:
            return '__end__'

    def analyzer_node(self, state: GenerateAgentState) -> Command[Literal['generator', 'generator_router']]:
        tool_call: ToolCall = state['messages'][-1].tool_calls[0]
        tool_call_id = tool_call['id']

        with Session(engine) as session:
            drafts = get_drafts(session, state['project_uid'])

        reranker = load_reranker()
        clean_drafts = reranker.compress_manuscripts(drafts, state['write_request'])

        if clean_drafts:
            state['messages'].pop(-1)
            return Command(
                update={
                    'messages': [
                        AIMessage(
                            '', tool_calls=[
                                ToolCall(
                                    name=self.generator.name,
                                    args={},
                                    id=tool_call_id
                                )
                            ]
                        )
                    ],
                    'information_list': clean_drafts
                },
                goto='generator'
            )

        return Command(
            update={
                'messages': [ToolMessage('本地没有这方面的资料，请你提供给我一些对于这个写作有帮助的资料', tool_call_id=tool_call_id)]
            },
            goto='generator_router'
        )

    def generator_node(self, state: GenerateAgentState):
        tool_call: ToolCall = state['messages'][-1].tool_calls[0]
        tool_call_id = tool_call['id']

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(GENERATOR_SYSTEM_ZH),
            ('human', GENERATOR_HUMAN_ZH)
        ])

        chain = prompt | self.task_llm
        response = chain.invoke({
            'question': state['write_request'],
            'information': '\n\n'.join([
                draft.content
                for draft in state['information_list']
            ])
        })

        return {
            'messages': [ToolMessage('我已经完成了写作任务', tool_call_id=tool_call_id)],
            'current_text': response.content
        }

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(GenerateAgentState)
        graph.add_node('generator_router', self.generator_route_node)
        graph.add_node('analyzer', self.analyzer_node)
        graph.add_node('generator', self.generator_node)

        graph.add_edge(START, 'generator_router')
        graph.add_conditional_edges('generator_router', self.generator_route)
        graph.add_edge('generator', 'generator_router')

        return graph.compile()


class MainAgentState(BaseAgentState):
    current_text: str
    chat_history: Annotated[list[AnyMessage], add_messages]
    sources: Annotated[list[dict[str, Document]], operator.add]


class MainAgent(BaseModel):
    router_llm: Optional[ChatOpenAI] = None
    task_llm: Optional[ChatOpenAI] = None
    use_web: bool = False

    @model_validator(mode='after')
    def setup_llm(self):
        if self.router_llm is None:
            self.router_llm = load_gpt4o()

        if self.task_llm is None:
            self.task_llm = load_gpt4o_mini()

    @staticmethod
    @tool
    def generate_task():
        """如果你需要从头开始撰写全新的文本内容，请调用此工具。
        它能够基于主题、关键词从头生成原创内容，适用于文章、故事、报告等各类写作需求。
        """
        return

    @staticmethod
    @tool
    def optimize_task():
        """如果你需要在现成的文本基础上进行修改、润色、优化或重构，请调用此工具。
        这个工具不会再引入新的知识，只会忠实的利用现有的文本进行修改。
        它将帮助你精细调整句式、语法、结构，提升文本的清晰度、流畅度和表达效果，适合需要修改、修订或增强已有文本的任务。
        """
        return

    @staticmethod
    @tool
    def search_task():
        """如果你需要搜索新的信息用于写作，请调用此工具。
        它能够从向量数据库、网络等渠道根据你的需求查询信息，并将结果汇总后返回以用于进一步的写作任务。
        """
        return

    def main_route_node(
            self, state: MainAgentState
    ):
        tools = [self.generate_task, self.optimize_task, self.search_task]
        llm_with_tool = self.router_llm.bind_tools(tools)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=AGENT_SYSTEM_ZH),
            MessagesPlaceholder(variable_name='history')
        ])

        # 所有对话以 chat_history 传入
        if not state['messages']:
            user_question = state['chat_history'][-1]
            assert isinstance(HumanMessage, user_question)

            state['messages'] = [user_question]

        chain = prompt | llm_with_tool
        response = chain.invoke({'history': state['messages']})
        token_usage = UsageMetadata.create(response.usage_metadata)

        return {
            'messages': [response],
            'price': token_usage.calculate_cost(self.router_llm)
        }

    def main_route(self, state: MainAgentState) -> Literal['text_generator', 'text_optimizer', 'knowledge_searcher', '__end__']:
        message: AIMessage = state['messages'][-1]

        if message.tool_calls:
            tool_call = message.tool_calls[0]

            if tool_call['name'] == self.generate_task.name:
                return 'text_generator'

            if tool_call['name'] == self.optimize_task.name:
                return 'text_optimizer'

            if tool_call['name'] == self.search_task.name:
                return 'knowledge_searcher'

        return '__end__'

    def text_generator_agent(self, state: MainAgentState):
        message: AIMessage = state['messages'][-1]
        tool_call = message.tool_calls[0]
        tool_call_id = tool_call['id']

        subgraph = GenerateAgent(
            router_llm=self.router_llm,
            task_llm=self.task_llm
        ).build()

        message = trim_messages(
            state['messages'],
            strategy='last',
            token_counter=len,
            max_tokens=5,
            start_on='human',
            end_on='human',
            include_system=False
        )

        output: GenerateAgentState = subgraph.invoke({
            'messages': message,
            'project_uid': state['project_uid']
        })
        return {
            'messages': [ToolMessage(output['messages'][-1].content, tool_call_id=tool_call_id)],
            'current_text': output['current_text'],
            'price': output['price']
        }

    def knowledge_searcher_agent(self, state: MainAgentState):
        message: AIMessage = state['messages'][-1]
        tool_call = message.tool_calls[0]
        tool_call_id = tool_call['id']

        subgraph = KnowledgeManageAgent(
            router_llm=self.router_llm,
            task_llm=self.task_llm,
            use_web=self.use_web
        ).build()

        message = trim_messages(
            state['messages'],
            strategy='last',
            token_counter=len,
            max_tokens=5,
            start_on='human',
            end_on=('human', 'tool'),
            include_system=False
        )
        output: KnowledgeManageAgentState = subgraph.invoke({
            'messages': [HumanMessage(content=message[-1].content)],
            'project_uid': state['project_uid']
        })

        new_sources = []
        for doc in output['documents']:
            meta_date = MarkdownMeta.model_validate(doc.metadata)
            if not str(meta_date.source) in state['sources']:
                new_sources.append({
                    str(meta_date.source): doc
                })

        return {
            'messages': [ToolMessage(output['messages'][-1].content, tool_call_id=tool_call_id)],
            'sources': new_sources,
            'price': output['price']
        }

    def text_optimizer_agent(self, state: MainAgentState):
        message: AIMessage = state['messages'][-1]
        tool_call = message.tool_calls[0]
        tool_call_id = tool_call['id']

        subgraph = OptimizerAgent(
            router_llm=self.router_llm,
            task_llm=self.task_llm
        ).build()
        message = trim_messages(
            state['messages'],
            strategy='last',
            token_counter=len,
            max_tokens=5,
            start_on='human',
            end_on='human',
            include_system=False
        )
        response = subgraph.invoke({
            'messages': message,
            'current_text': state['current_text']
        })
        return {
            'messages': [ToolMessage(content=response['messages'][-1].content, tool_call_id=tool_call_id)],
            'current_text': response['current_text'],
            'price': response['price']
        }

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(MainAgentState)
        graph.add_node('main_router', self.main_route_node)
        graph.add_node('text_generator', self.text_generator_agent)
        graph.add_node('text_optimizer', self.text_optimizer_agent)
        graph.add_node('knowledge_searcher', self.knowledge_searcher_agent)

        graph.add_edge(START, 'main_router')
        graph.add_conditional_edges('main_router', self.main_route)
        graph.add_edge('text_optimizer', 'main_router')
        graph.add_edge('knowledge_searcher', 'main_router')
        graph.add_edge('text_generator', 'main_router')

        return graph.compile()

    def visualize(self, file_path: str) -> None:
        graph = StateGraph(MainAgentState)
        graph.add_node('main_router', self.main_route_node)
        graph.add_node('text_generator', GenerateAgent(router_llm=self.router_llm, task_llm=self.task_llm).build())
        graph.add_node('text_optimizer', OptimizerAgent(router_llm=self.router_llm, task_llm=self.task_llm).build())
        graph.add_node('knowledge_searcher', KnowledgeManageAgent(router_llm=self.router_llm, task_llm=self.task_llm).build())

        graph.add_edge(START, 'main_router')
        graph.add_conditional_edges('main_router', self.main_route)
        graph.add_edge('text_optimizer', 'main_router')
        graph.add_edge('knowledge_searcher', 'main_router')
        graph.add_edge('text_generator', 'main_router')

        temp_app = graph.compile()
        temp_app.get_graph(xray=True).draw_mermaid_png(
            output_file_path=file_path,
            draw_method=MermaidDrawMethod.PYPPETEER,
            curve_style=CurveStyle.BASIS
        )
