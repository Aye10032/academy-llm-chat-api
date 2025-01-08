from operator import itemgetter
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableSerializable

from llm.core.model import load_gpt4o_mini
from llm.core.template import RAG_SYSTEM_EN, RAG_HUMAN_EN
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
    llm = load_gpt4o_mini()
    chain = prompt | llm

    result = chain.invoke({
        'chat_history': chat_history,
        'input': question
    })

    return result


def rag_chat(
        question: str,
        docs: list[Document],
        *,
        chat_history: Optional[list[BaseMessage]] = None
) -> RunnableSerializable:
    if chat_history is None:
        chat_history = []

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=RAG_SYSTEM_EN),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', RAG_HUMAN_EN),
        ]
    )
    llm = load_gpt4o_mini()
    formatter = itemgetter("docs") | RunnableLambda(format_docs)
    chain = {
                'chat_history': itemgetter('chat_history'),
                'documents': formatter,
                'question': itemgetter('question')
            } | prompt | llm

    return chain


def conclude_chat(_chat_history: BaseChatMessageHistory):
    """
    Summarize the main content of a chat conversation.

    This function generates a summary of the chat conversation, including the identities of the participants,
    the topics discussed, key points raised, and any questions or solutions proposed. The summary is concise,
    not exceeding 18 characters.

    :param _chat_history: The chat history to be summarized.
    :return: A summary of the chat conversation.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="你即将看到一段对话记录。"
                        "请总结对话的主要内容，包括对话参与者的身份、讨论的主题、提出的关键观点、问题或解决方案。"
                        "确保抓住对话中的重要细节和关键时刻，同时控制字数，不要超过18字。"
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(
                content="根据对话内容，生成一个词组或短语（**不超过18个字**）作为该对话的概览词。该概览词应能反映对话的核心主题或目的。"),
        ]
    )
    llm = load_gpt4o_mini()
    chain = prompt | llm

    result = chain.invoke({"history": _chat_history.messages})

    return result
