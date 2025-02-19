from operator import itemgetter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableSerializable

from llm.core.model import load_llm
from llm.core.template import RAG_SYSTEM_ZH, RAG_HUMAN_ZH
from llm.rag.retriever import format_docs


def simple_chat(question: str, *, chat_history: list[BaseMessage]):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                'You are a helpful assistant. Answer all questions to the best of your ability.',
            ),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{input}'),
        ]
    )
    llm = load_llm('gpt-4o-mini')
    chain = prompt | llm

    result = chain.invoke({'chat_history': chat_history, 'input': question})

    return result


def rag_chain(model_name: str, temperature: float) -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=RAG_SYSTEM_ZH),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', RAG_HUMAN_ZH),
        ]
    )

    llm = load_llm(model_name, temperature)
    formatter = itemgetter('docs') | RunnableLambda(format_docs)
    chain = (
        {
            'chat_history': itemgetter('chat_history'),
            'documents': formatter,
            'question': itemgetter('question'),
        }
        | prompt
        | llm
    )

    return chain


def conclude_chat(chat_history: BaseChatMessageHistory):
    """对对话的上下文进行总结

    Args:
        chat_history: 待总结的对话历史记录

    Returns:
        18个字符以内的关于对话的总结
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content='你即将看到一段对话记录。'
                '请总结对话的主要内容，包括对话参与者的身份、讨论的主题、提出的关键观点、问题或解决方案。'
                '确保抓住对话中的重要细节和关键时刻，同时控制字数，不要超过18字。'
            ),
            MessagesPlaceholder(variable_name='history'),
            HumanMessage(
                content='根据对话内容，生成一个词组或短语（**不超过18个字**）作为该对话的概览词。该概览词应能反映对话的核心主题或目的。'
            ),
        ]
    )
    llm = load_llm('gpt-4o-mini')
    chain = prompt | llm
    short_message = trim_messages(
        chat_history.messages,
        strategy='last',
        token_counter=len,
        max_tokens=5,
        start_on='human',
        end_on=('ai', 'tool'),
        include_system=False,
    )
    result = chain.invoke({'history': short_message})

    return result
