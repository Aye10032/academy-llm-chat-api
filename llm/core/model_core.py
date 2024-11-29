from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from llm.core.embedding_core import BgeM3Embeddings, BgeReranker

embd_cfg = get_settings().retriever.embedding
reranker_cfg = get_settings().retriever.reranker
llm_cfg = get_settings().llm


def load_embedding() -> BgeM3Embeddings:
    embedding = BgeM3Embeddings(
        bge_model_name=embd_cfg.MODEL,
        use_fp16=embd_cfg.FP16,
        device=embd_cfg.DEVICE,
        encode_kwargs={
            'normalize_embeddings': embd_cfg.NORMALIZE
        },
        local_load=embd_cfg.SAVE_LOCAL,
        local_path=embd_cfg.LOCAL_PATH
    )

    return embedding


def load_reranker() -> BgeReranker:
    reranker = BgeReranker(
        bge_model_name=reranker_cfg.MODEL,
        use_fp16=reranker_cfg.FP16,
        device=reranker_cfg.DEVICE,
        encode_kwargs={
            'normalize': reranker_cfg.NORMALIZE
        },
        local_load=reranker_cfg.SAVE_LOCAL,
        local_path=reranker_cfg.LOCAL_PATH
    )

    return reranker


def load_gpt4o() -> ChatOpenAI:
    if llm_cfg.openai.USE_PROXY:
        llm = ChatOpenAI(
            model_name='gpt-4o',
            openai_proxy=get_settings().PROXY,
            temperature=0.4,
            openai_api_key=llm_cfg.openai.API_KEY
        )
    else:
        llm = ChatOpenAI(
            model_name='gpt-4o',
            temperature=0.4,
            openai_api_key=llm_cfg.openai.API_KEY
        )
    return llm


def load_gpt4o_mini() -> ChatOpenAI:
    if llm_cfg.openai.USE_PROXY:
        llm = ChatOpenAI(
            model_name='gpt-4o-mini',
            openai_proxy=get_settings().PROXY,
            temperature=0.4,
            openai_api_key=llm_cfg.openai.API_KEY
        )
    else:
        llm = ChatOpenAI(
            model_name='gpt-4o-mini',
            temperature=0.4,
            openai_api_key=llm_cfg.openai.API_KEY
        )
    return llm


def load_glm4_flash() -> ChatOpenAI:
    llm = ChatOpenAI(
        model_name='glm-4-flash',
        openai_api_base=llm_cfg.zhipu.BASE_URL,
        openai_api_key=llm_cfg.zhipu.API_KEY,
        temperature=0.05,
    )

    return llm
