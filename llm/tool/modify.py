from typing import Any, Optional, Type

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import model_validator, BaseModel, Field

from llm.core.model import load_gpt4o
from llm.core.template import MODIFY_SYSTEM_ZH, REWRITER_SYSTEM_ZH, OPTIMIZER_HUMAN_ZH


class RewriterInput(BaseModel):
    query: str = Field(description='具体的修改需求')
    current_text: str = Field(description='待修改的原始文本，这会在后续调用中由使用者手动给出，你只需返回空字符占位即可', default='')


class RewriterOutput(BaseModel):
    rewrite: str = Field(description='重写后的完整文本')
    explanation: str = Field(
        description='你对于此次修改工作的总结。仅需简要说明本次优化的主要方向，如语言风格、结构调整、专业术语使用等，而无需列出具体的修改内容')


class Rewriter(BaseTool):
    name: str = 'rewriter'
    description: str = '我负责对文本进行整体性重构，包括但不限于结构调整、段落重组、逻辑重塑、内容扩展/压缩等全局性优化'
    args_schema: Type[BaseModel] = RewriterInput
    return_direct: bool = False
    handle_tool_error: bool = True

    llm: Optional[ChatOpenAI] = None

    @model_validator(mode='after')
    def init_llm(self):
        if self.llm is None:
            self.llm = load_gpt4o()

        return self

    def _run(self, query: str, current_text: str, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=REWRITER_SYSTEM_ZH),
            ('human', OPTIMIZER_HUMAN_ZH)
        ])
        chain = prompt | self.llm.with_structured_output(RewriterOutput, method='function_calling', include_raw=True)

        return chain.invoke({
            'origin_text': current_text,
            'question': query
        }, config)


class ModifierInput(BaseModel):
    query: str = Field(description='具体的修改需求')
    current_text: str = Field(description='待修改的原始文本，这会在后续调用中由使用者手动给出，你只需返回空字符占位即可', default='')


class Modification(BaseModel):
    """具体的修改内容，对于每一处修改均需要给出简单的修改理由"""
    original: str = Field(description='原文中需要修改的原句')
    modified: str = Field(description='修改后的句子')
    explanation: str = Field(description='做出此修改的原因')


class OptimizerOutput(BaseModel):
    """你对于原始文本的改动意见"""
    modifies: list[Modification] = Field(description='修改意见列表，其中每一条修改意见都需要满足规定的格式')


class Modifier(BaseTool):
    name: str = 'modifier'
    description: str = '我专注局部优化，包括词语替换、句式调整、语法修正、标点规范等细节修改'
    args_schema: Type[BaseModel] = ModifierInput
    return_direct: bool = False
    handle_tool_error: bool = True

    llm: Optional[ChatOpenAI] = None

    @model_validator(mode='after')
    def init_llm(self):
        if self.llm is None:
            self.llm = load_gpt4o()

        return self

    def _run(self, query: str, current_text: str, config: Optional[RunnableConfig] = None) -> Any:
        llm = self.llm.with_structured_output(OptimizerOutput, include_raw=True, method='function_calling')
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=MODIFY_SYSTEM_ZH),
            ('human', OPTIMIZER_HUMAN_ZH)
        ])
        chain = prompt | llm

        return chain.invoke({
            'origin_text': current_text,
            'question': query
        }, config)
