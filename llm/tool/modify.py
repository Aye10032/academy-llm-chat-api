from typing import Optional, Type

from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import model_validator, BaseModel, Field

from llm.core.model import load_llm
from llm.core.template import MODIFY_SYSTEM_ZH, REWRITER_SYSTEM_ZH, OPTIMIZER_HUMAN_ZH


class RewriterInput(BaseModel):
    query: str = Field(description='具体的修改需求')
    current_text: str = Field(
        description='待修改的原始文本，这会在后续调用中由使用者手动给出，你只需返回空字符占位即可',
        default='',
    )


class Rewriter(BaseTool):
    name: str = 'rewriter'
    description: str = '我负责对文本进行整体性重构，包括但不限于结构调整、段落重组、逻辑重塑、内容扩展/压缩、文本翻译等全局性改动'
    args_schema: Type[BaseModel] = RewriterInput
    return_direct: bool = False
    handle_tool_error: bool = True

    llm: Optional[ChatOpenAI] = None

    @model_validator(mode='after')
    def init_llm(self):
        if self.llm is None:
            self.llm = load_llm('gpt-4o-mini')

        return self

    def _run(
        self, query: str, current_text: str, config: Optional[RunnableConfig] = None
    ) -> BaseMessage:
        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=REWRITER_SYSTEM_ZH), ('human', OPTIMIZER_HUMAN_ZH)]
        )
        chain = prompt | self.llm

        return chain.invoke({'origin_text': current_text, 'question': query}, config)


class ModifierInput(BaseModel):
    query: str = Field(description='具体的修改需求')
    current_text: str = Field(
        description='待修改的原始文本，这会在后续调用中由使用者手动给出，你只需返回空字符占位即可',
        default='',
    )


class Modification(BaseModel):
    """具体的修改内容，对于每一处修改均需要给出简单的修改理由"""

    original: str = Field(description='原文中需要修改的原句')
    modified: str = Field(description='修改后的句子')
    explanation: str = Field(description='做出此修改的原因')


class OptimizerOutput(BaseModel):
    """你对于原始文本的改动意见"""

    modifies: list[Modification] = Field(
        description='修改意见列表，其中每一条修改意见都需要满足规定的格式'
    )


class Modifier(BaseTool):
    name: str = 'modifier'
    description: str = (
        '我专注局部优化，包括词语替换、句式调整、语法修正、标点规范等细节修改'
    )
    args_schema: Type[BaseModel] = ModifierInput
    return_direct: bool = False
    handle_tool_error: bool = True

    llm: Optional[ChatOpenAI] = None

    @model_validator(mode='after')
    def init_llm(self):
        if self.llm is None:
            self.llm = load_llm('gpt-4o')

        return self

    def _run(
        self, query: str, current_text: str, config: Optional[RunnableConfig] = None
    ) -> OptimizerOutput:
        prompt = ChatPromptTemplate.from_messages(
            [('system', MODIFY_SYSTEM_ZH), ('human', OPTIMIZER_HUMAN_ZH)]
        )
        parser = PydanticOutputParser(pydantic_object=OptimizerOutput)
        chain = prompt | self.llm | parser

        return chain.invoke(
            {
                'origin_text': current_text,
                'question': query,
                'structure': parser.get_format_instructions(),
            },
            config,
        )
