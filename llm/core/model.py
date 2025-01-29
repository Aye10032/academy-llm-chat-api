import json
import os
import re
from typing import Any, Optional, Union, Sequence

import httpx
import numpy as np
import torch
import transformers.utils.logging
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field
from torch import Tensor
from tqdm import tqdm

from app.core.config import get_settings
from app.utils.cache import cache_model

embd_cfg = get_settings().retriever.embedding
reranker_cfg = get_settings().retriever.reranker
jina_cfg = get_settings().tool.jina
llm_cfg = get_settings().llm


class BgeM3Embeddings(BaseModel, Embeddings):
    bge_model_name: str = 'BAAI/bge-m3'
    bge_tokenizer: Any = None
    bge_model: Any = None

    """
    Keyword arguments to pass to the model.

    pooling_method: str = 'cls',
    use_fp16: bool = True,
    device: str = None
    """
    pooling_method: str = 'cls'
    use_fp16: bool = True
    device: Optional[str] = None

    """
    Keyword arguments to pass when calling the `encode` method of the model.

    normalize_embeddings: bool = True,
    batch_size: int = 12,
    max_length: int = 8192,
    """
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)

    local_load: bool = False
    local_path: str = ''

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            from transformers import (
                AutoTokenizer,
                AutoModel,
                PreTrainedTokenizerFast,
                PreTrainedModel,
            )

        except ImportError as exc:
            raise ImportError(
                'Could not import transformers python package. '
                'Please install it with `pip install transformers`.'
            ) from exc

        if self.local_load:
            try:
                self.bge_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.local_path)
                self.bge_model: PreTrainedModel = AutoModel.from_pretrained(self.local_path)
            except EnvironmentError:
                logger.warning('Load model from local fail. Download from huggingface...')

                self.bge_tokenizer = AutoTokenizer.from_pretrained(self.bge_model_name)
                self.bge_model = AutoModel.from_pretrained(self.bge_model_name)

                # save to local
                os.makedirs(self.local_path, exist_ok=True)
                self.bge_tokenizer.save_pretrained(self.local_path)
                self.bge_model.save_pretrained(self.local_path)
        else:
            self.bge_tokenizer = AutoTokenizer.from_pretrained(self.bge_model_name)
            self.bge_model = AutoModel.from_pretrained(self.bge_model_name)

        if not self.device:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if not torch.cuda.is_available():
            self.use_fp16 = False

        if self.use_fp16:
            self.bge_model.half()

        self.bge_model = self.bge_model.to(torch.device(self.device))
        self.bge_model.eval()

    def dense_embedding(self, hidden_state: Tensor, mask: Tensor) -> Tensor:
        if self.pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(dim=1, keepdim=True).float()
            return s / d

    @torch.no_grad()
    def encode(
            self,
            sentences: Union[list[str], str],
            normalize_embeddings: bool = True,
            batch_size: int = 12,
            max_length: int = 8192,
    ) -> np.ndarray:
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_dense_embeddings = []
        for start_index in tqdm(
                range(0, len(sentences), batch_size),
                desc='Inference Embeddings',
                disable=len(sentences) < 256
        ):
            sentences_batch = sentences[start_index:start_index + batch_size]
            batch_data = self.bge_tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)

            last_hidden_state = self.bge_model(**batch_data, return_dict=True).last_hidden_state
            dense_vecs = self.dense_embedding(last_hidden_state, batch_data['attention_mask'])

            if normalize_embeddings:
                dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)
            all_dense_embeddings.append(dense_vecs.cpu().numpy())

        all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)
        if input_was_string:
            all_dense_embeddings = all_dense_embeddings[0]

        return all_dense_embeddings

    def embed_query(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        embedding = self.encode(text, **self.encode_kwargs)
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self.encode(texts, **self.encode_kwargs)

        return embeddings.tolist()


class BgeReranker(BaseModel):
    bge_model_name: str = 'BAAI/bge-reranker-v2-m3'
    bge_tokenizer: Any = None
    bge_model: Any = None

    """
    Keyword arguments to pass to the model.

    use_fp16: bool = False,
    device: Union[str, int] = None
    """
    use_fp16: bool = False,
    device: Optional[str] = None

    """
    Keyword arguments to pass when calling the `compress_documents` method of the model.

    batch_size: int = 256,
    max_length: int = 512, 
    normalize: bool = False
    """
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)

    local_load: bool = False
    local_path: str = ''

    drop_low_score: bool = True
    low_score: float = 0.1

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                PreTrainedTokenizerFast,
                PreTrainedModel,
            )

        except ImportError as exc:
            raise ImportError(
                'Could not import transformers python package. '
                'Please install it with `pip install transformers`.'
            ) from exc

        if self.local_load:
            try:
                self.bge_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.local_path)
                self.bge_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(self.local_path)
            except EnvironmentError:
                logger.warning('Load model from local fail. Download from huggingface...')

                self.bge_tokenizer = AutoTokenizer.from_pretrained(self.bge_model_name)
                self.bge_model = AutoModelForSequenceClassification.from_pretrained(self.bge_model_name)

                # save to local
                os.makedirs(self.local_path, exist_ok=True)
                self.bge_tokenizer.save_pretrained(self.local_path)
                self.bge_model.save_pretrained(self.local_path)
        else:
            self.bge_tokenizer = AutoTokenizer.from_pretrained(self.bge_model_name)
            self.bge_model = AutoModelForSequenceClassification.from_pretrained(self.bge_model_name)

        if not self.device:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if not torch.cuda.is_available():
            self.use_fp16 = False

        if self.use_fp16:
            self.bge_model.half()

        self.bge_model = self.bge_model.to(torch.device(self.device))
        self.bge_model.eval()

    @torch.no_grad()
    def compute_score(
            self,
            sentence_pairs: Union[list[tuple[str, str]], tuple[str, str]],
            batch_size: int = 256,
            max_length: int = 512,
            normalize: bool = False
    ) -> list[float]:

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        all_scores = []
        for start_index in tqdm(
                range(0, len(sentence_pairs), batch_size),
                desc='Compute Scores',
                disable=len(sentence_pairs) < 128
        ):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            inputs = self.bge_tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)

            scores = self.bge_model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy().tolist())

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        return all_scores

    def compress_documents(
            self,
            documents: list[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> list[Document]:

        sentence_pairs = [
            (query, doc.page_content)
            for doc in documents
            if isinstance(doc, Document)
        ]
        rerank_scores = []

        for i in range(0, len(sentence_pairs), 10):
            if i + 10 >= len(sentence_pairs):
                batch_pairs = sentence_pairs[i:]
            else:
                batch_pairs = sentence_pairs[i:i + 10]

            batch_scores = self.compute_score(batch_pairs, **self.encode_kwargs)
            rerank_scores.extend(batch_scores)

        rerank_results = list(zip(rerank_scores, documents))
        rerank_results = sorted(rerank_results, key=lambda x: x[0], reverse=True)

        final_results = [
            doc for r in rerank_results
            if (doc := r[1]).metadata.update({'score': r[0]}) or (not self.drop_low_score or r[0] > self.low_score)
        ]
        return final_results

    async def acompress_documents(
            self,
            documents: list[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""
        return await run_in_executor(
            None, self.compress_documents, documents, query, callbacks
        )


# Patterns
SCRIPT_PATTERN = r'<[ ]*script.*?\/[ ]*script[ ]*>'
STYLE_PATTERN = r'<[ ]*style.*?\/[ ]*style[ ]*>'
META_PATTERN = r'<[ ]*meta.*?>'
COMMENT_PATTERN = r'<[ ]*!--.*?--[ ]*>'
LINK_PATTERN = r'<[ ]*link.*?>'
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r'(<svg[^>]*>)(.*?)(<\/svg>)'


class ReaderLM(BaseModel):
    jina_model_name: str
    jina_tokenizer: Any = None
    jina_model: Any = None

    """
    Keyword arguments to pass to the model.

    use_fp16: bool = False,
    device: Union[str, int] = None
    """
    use_fp16: bool = False,
    device: Optional[str] = None

    """
    Keyword arguments to pass when calling the `compress_documents` method of the model.

    batch_size: int = 256,
    max_length: int = 512, 
    normalize: bool = False
    """
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)

    local_load: bool = False
    local_path: str = ''

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                PreTrainedTokenizerFast,
                PreTrainedModel,
            )

        except ImportError as exc:
            raise ImportError(
                'Could not import transformers python package. '
                'Please install it with `pip install transformers`.'
            ) from exc

        transformers.utils.logging.enable_progress_bar()

        if self.local_load:
            try:
                self.jina_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.local_path)
                self.jina_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(self.local_path)
            except EnvironmentError:
                logger.warning('Load model from local fail. Download from huggingface...')

                logger.debug(self.jina_model_name)
                self.jina_tokenizer = AutoTokenizer.from_pretrained(self.jina_model_name, force_download=True)
                self.jina_model = AutoModelForCausalLM.from_pretrained(self.jina_model_name, force_download=True)

                # save to local
                os.makedirs(self.local_path, exist_ok=True)
                self.jina_tokenizer.save_pretrained(self.local_path)
                self.jina_model.save_pretrained(self.local_path)
        else:
            self.jina_tokenizer = AutoTokenizer.from_pretrained(self.jina_model_name)
            self.jina_model = AutoModelForCausalLM.from_pretrained(self.jina_model_name)

        if not self.device:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if not torch.cuda.is_available():
            self.use_fp16 = False

        if self.use_fp16:
            self.jina_model.half()

        self.jina_model = self.jina_model.to(torch.device(self.device))
        self.jina_model.eval()

    @staticmethod
    def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
        return re.sub(
            SVG_PATTERN,
            lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
            html,
            flags=re.DOTALL,
        )

    @staticmethod
    def replace_base64_images(html: str, new_image_src: str = "#") -> str:
        return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)

    def clean_html(self, html: str, clean_svg: bool = False, clean_base64: bool = False):
        html = re.sub(
            SCRIPT_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            STYLE_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            META_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            COMMENT_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            LINK_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )

        if clean_svg:
            html = self.replace_svg(html)
        if clean_base64:
            html = self.replace_base64_images(html)
        return html

    @staticmethod
    def create_prompt(
            text: str, tokenizer=None, instruction: str = None, schema: str = None
    ) -> str:
        """
        Create a prompt for the model with optional instruction and JSON schema.
        """
        if not instruction:
            instruction = 'Extract the main content from the given HTML and convert it to Markdown format.'
        if schema:
            instruction = 'Extract the specified information from a list of news threads and present it in a structured JSON format.'
            prompt = f'{instruction}\n```html\n{text}\n```\nThe JSON schema is as follows:```json\n{schema}\n```'
        else:
            prompt = f'{instruction}\n```html\n{text}\n```'

        messages = [
            {
                'role': 'user',
                'content': prompt,
            }
        ]

        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def html_to_md(self, html: str) -> str:

        html = self.clean_html(html)

        input_prompt = self.create_prompt(html, tokenizer=self.jina_tokenizer)
        inputs = self.jina_tokenizer.encode(input_prompt, return_tensors='pt').to(self.device)
        outputs = self.jina_model.generate(
            inputs, max_new_tokens=1024, temperature=0, do_sample=False, repetition_penalty=1.08
        )

        return self.jina_tokenizer.decode(outputs[0])

    def html_to_json(self, html: str) -> str:
        schema = """
        {
          "type": "object",
          "properties": {
            "title": {
              "type": "string"
            },
            "author": {
              "type": "string"
            },
            "date": {
              "type": "string"
            },
            "content": {
              "type": "string"
            }
          },
          "required": ["title", "author", "date", "content"]
        }
        """

        html = self.clean_html(html)
        input_prompt = self.create_prompt(html, tokenizer=self.jina_tokenizer, schema=schema)

        inputs = self.jina_tokenizer.encode(input_prompt, return_tensors='pt').to(self.device)
        outputs = self.jina_model.generate(
            inputs, max_new_tokens=1024, do_sample=False, repetition_penalty=1.08
        )
        json_text = self.jina_tokenizer.decode(outputs[0])
        return json.loads(json_text)


@cache_model()
def load_embedding() -> BgeM3Embeddings:
    logger.info(f'加载Embedding模型 {embd_cfg.MODEL}...')
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


@cache_model()
def load_reranker() -> BgeReranker:
    logger.info(f'加载Rerank模型 {reranker_cfg.MODEL}...')
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


@cache_model()
def load_jina_reader() -> ReaderLM:
    logger.info(f'加载html读取模型 {jina_cfg.MODEL}...')

    reader = ReaderLM(
        jina_model_name=jina_cfg.MODEL,
        use_fp16=jina_cfg.FP16,
        device=jina_cfg.DEVICE,
        local_load=jina_cfg.SAVE_LOCAL,
        local_path=jina_cfg.LOCAL_PATH
    )

    return reader


def load_gpt4o() -> ChatOpenAI:
    if llm_cfg.openai.USE_PROXY:
        http_client = httpx.Client(proxy=get_settings().server.network.PROXY)
        llm = ChatOpenAI(
            model_name='gpt-4o',
            http_client=http_client,
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
            openai_proxy=get_settings().server.network.PROXY,
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


def load_glm4_air() -> ChatOpenAI:
    llm = ChatOpenAI(
        model_name='glm-4-air',
        openai_api_base=llm_cfg.zhipu.BASE_URL,
        openai_api_key=llm_cfg.zhipu.API_KEY,
        temperature=0.05,
    )

    return llm


def load_deepseek_v3(temperature: float = 0.0) -> ChatOpenAI:
    llm = ChatOpenAI(
        model_name='deepseek-chat',
        openai_api_base=llm_cfg.deepseek.BASE_URL,
        openai_api_key=llm_cfg.deepseek.API_KEY,
        temperature=temperature
    )

    return llm


def load_deepseek_r1() -> ChatOpenAI:
    llm = ChatOpenAI(
        model_name='deepseek-reasoner',
        openai_api_base=llm_cfg.deepseek.BASE_URL,
        openai_api_key=llm_cfg.deepseek.API_KEY
    )

    return llm
