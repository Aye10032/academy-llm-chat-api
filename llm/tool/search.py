import json
from typing import Type

import requests
from duckduckgo_search import DDGS
from langchain_core.tools import BaseTool, ToolException
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import get_settings

config = get_settings()


class WebSearchInput(BaseModel):
    query: str = Field(description='联网搜索的关键词')


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

    def search(self, query: str):
        if serper_api := config.tool.search.SERPER_API:
            url = 'https://google.serper.dev/search'

            payload = json.dumps({
                'q': query,
                'k': self.max_search_result,
                'gl': 'us',
                'hl': 'en'
            })
            headers = {
                'X-API-KEY': serper_api,
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            if config.server.network.USE_PROXY:
                response = requests.request(
                    'POST', url,
                    headers=headers,
                    data=payload,
                    proxies={
                        'http': config.server.network.PROXY,
                        'https': config.server.network.PROXY,
                    },
                    timeout=60
                )
            else:
                response = requests.request(
                    'POST', url,
                    headers=headers,
                    data=payload,
                    timeout=60
                )

            if response.status_code == 200:
                search_data = json.loads(response.text)
                print(search_data)
                url_list = [
                    organic['link']
                    for organic in search_data['organic']
                ]
                return url_list
        else:
            logger.warning('no serper api found, using ddgs instead')
            if config.server.network.USE_PROXY:
                with DDGS(proxy=config.server.network.PROXY) as ddgs:
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
                print(search_result)
                url_list = [
                    result['href']
                    for result in search_result
                ]
                return url_list

        raise ToolException('所给出的问题没有在互联网上找到相关信息。')

    def _run(self, query: str) -> list[str]:
        """调用工具进行联网搜索"""
        logger.info(f'Calling WebSearchTool with query {query}')

        urls = self.search(query)
        return urls
