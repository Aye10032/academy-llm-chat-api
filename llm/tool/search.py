import json
from typing import Optional, Type

import requests
from duckduckgo_search import DDGS
from langchain_core.tools import BaseTool, ToolException
from loguru import logger
from pydantic import AnyHttpUrl, BaseModel, Field

from app.core.config import get_settings


class WebSearchInput(BaseModel):
    question: str = Field(description='联网搜索的关键词')


class WebSearchResult(BaseModel):
    title: str
    source: AnyHttpUrl
    description: Optional[str] = ''


class WebSearchTool(BaseTool):
    """联网搜索工具

    此工具主要任务是返回搜寻结果的url，具体的内容解读请结合WebLoader使用
    """

    name: str = 'search_from_web'
    description: str = '通过搜索引擎进行联网搜索。AI可以通过调用此工具，联网查询一些自己不清楚的或者比较新的信息。此工具只有在用户特别指明联网查询或者向量数据库搜索失败的情况下调用。'
    args_schema: Type[BaseModel] = WebSearchInput
    return_direct: bool = False
    handle_tool_error: bool = True

    region: str = 'wt-wt'
    max_search_result: int = 6

    def search(self, query: str) -> list[WebSearchResult]:
        if serper_api := get_settings().tool.search.SERPER_API:
            url = 'https://google.serper.dev/search'

            payload = json.dumps({'q': query, 'k': self.max_search_result, 'gl': 'us', 'hl': 'en'})
            headers = {
                'X-API-KEY': serper_api,
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            }

            if get_settings().network.USE_PROXY:
                response = requests.request(
                    'POST',
                    url,
                    headers=headers,
                    data=payload,
                    proxies={
                        'http': get_settings().network.PROXY,
                        'https': get_settings().network.PROXY,
                    },
                    timeout=60,
                )
            else:
                response = requests.request('POST', url, headers=headers, data=payload, timeout=60)

            if response.status_code == 200:
                search_data = json.loads(response.text)
                result_list = [
                    WebSearchResult(
                        title=organic['title'],
                        source=organic['link'],
                        description=organic['snippet'],
                    )
                    for organic in search_data['organic']
                ]
                return result_list
        else:
            logger.warning('no serper api found, using ddgs instead')
            if get_settings().network.USE_PROXY:
                with DDGS(proxy=get_settings().network.PROXY) as ddgs:
                    search_result = ddgs.text(
                        query,
                        region=self.region,
                        max_results=self.max_search_result,
                    )
            else:
                with DDGS() as ddgs:
                    search_result = ddgs.text(
                        query,
                        region=self.region,
                        max_results=self.max_search_result,
                    )

            if search_result:
                result_list = [
                    WebSearchResult(
                        title=result['title'],
                        source=result['href'],
                        description=result['body'],
                    )
                    for result in search_result
                ]
                return result_list

        raise ToolException('所给出的问题没有在互联网上找到相关信息。')

    def _run(self, question: str) -> list[WebSearchResult]:
        """调用工具进行联网搜索"""
        urls = self.search(question)
        return urls
