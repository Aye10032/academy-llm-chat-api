import asyncio
import os
from ftplib import FTP
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger
from tqdm.asyncio import tqdm


class FTPClient:
    """FTP客户端工具类，用于连接FTP服务器并执行文件操作"""

    def __init__(
        self, host: str, port: int = 21, username: str = '', password: str = '', timeout: int = 30
    ):
        """
        初始化FTP客户端

        Args:
            host: FTP服务器地址
            port: FTP服务器端口，默认为21
            username: 用户名，默认为空字符串（匿名登录）
            password: 密码，默认为空字符串
            timeout: 连接超时时间，默认为30秒
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        self.ftp = None
        self._is_connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self) -> bool:
        """
        连接到FTP服务器

        Returns:
            bool: 连接是否成功
        """
        try:
            self.ftp = FTP()
            self.ftp.connect(self.host, self.port, self.timeout)
            self.ftp.login(self.username, self.password)
            self._is_connected = True
            logger.info(f'成功连接到FTP服务器: {self.host}:{self.port}')
            return True
        except Exception as e:
            logger.error(f'连接FTP服务器失败: {str(e)}')
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """断开与FTP服务器的连接"""
        if self.ftp and self._is_connected:
            try:
                self.ftp.quit()
                logger.info(f'已断开与FTP服务器的连接: {self.host}:{self.port}')
            except Exception as e:
                logger.warning(f'断开FTP连接时发生异常: {str(e)}')
            finally:
                self._is_connected = False
                self.ftp = None

    def list_files(self, remote_path: str = '.') -> list[str]:
        """获取指定路径下的文件和目录列表

        Args:
            remote_path: 远程服务器上的路径，默认为当前目录

        Returns:
            文件名的列表
        """
        if not self._is_connected and not self.connect():
            logger.error('无法获取文件列表：未连接到FTP服务器')
            return []

        try:
            self.ftp.cwd(remote_path)

            file_list = []

            def process_line(line):
                parts = line.split()
                if len(parts) >= 9:
                    is_dir = parts[0].startswith('d')
                    name = ' '.join(parts[8:])
                    if name not in ('.', '..') and not is_dir:
                        file_list.append(name)

            self.ftp.retrlines('LIST', process_line)

            logger.info(f'成功获取目录 {remote_path} 下的文件列表，共 {len(file_list)} 个项目')
            return file_list

        except Exception as e:
            logger.error(f'获取文件列表失败: {str(e)}')
            return []


class HTTPDownloader:
    """HTTP文件下载器，支持异步并发下载"""

    def __init__(
        self,
        max_concurrency: int = 5,
        timeout: int = 30,
        chunk_size: int = 1024 * 1024,  # 1MB
    ):
        """
        初始化HTTP下载器

        Args:
            max_concurrency: 最大并发下载数，默认为5
            timeout: 连接超时时间，默认为30秒
            chunk_size: 下载分块大小，默认为1MB
        """
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.download_results: dict[str, bool] = {}
        self.failed_files: set[str] = set()
        self.session = None

        # 全局进度条
        self.global_pbar = None
        self.total_files = 0
        self.completed_files = 0

    async def _ensure_session(self):
        """确保aiohttp会话已创建"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

    async def download_file(
        self, url: str, local_path: str | os.PathLike, position: int = 0
    ) -> bool:
        """
        异步下载单个文件

        Args:
            url: 文件URL
            local_path: 本地保存路径
            position: 进度条位置，用于多文件下载时的进度条排列

        Returns:
            bool: 下载是否成功
        """
        async with self.semaphore:
            await self._ensure_session()

            os.makedirs(Path(local_path).parent, exist_ok=True)
            temp_path = f'{local_path}.download'

            try:
                async with self.session.get(url, allow_redirects=True) as response:
                    if response.status != 200:
                        logger.error(f'下载文件失败: {url}, 状态码: {response.status}')
                        # 更新全局进度条
                        if self.global_pbar:
                            self.completed_files += 1
                            self.global_pbar.update(1)
                        return False

                    file_size = int(response.headers.get('Content-Length', 0))
                    downloaded = 0

                    filename = Path(local_path).name
                    desc = f'下载 {filename}'

                    with open(temp_path, 'wb') as f:
                        with tqdm(
                            desc=desc,
                            total=file_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                            miniters=1,
                            position=position,
                            leave=False,
                        ) as pbar:
                            async for chunk in response.content.iter_chunked(self.chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    pbar.update(len(chunk))

                os.rename(temp_path, local_path)

                # 更新全局进度条
                if self.global_pbar:
                    self.completed_files += 1
                    self.global_pbar.update(1)

                return True

            except Exception as e:
                logger.error(f'下载文件异常 {url}: {str(e)}')
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # 更新全局进度条
                if self.global_pbar:
                    self.completed_files += 1
                    self.global_pbar.update(1)

                return False

    async def download_files(
        self, base_url: str, remote_path: str, file_list: list[str], local_dir: str
    ) -> dict[str, bool]:
        """批量下载文件

        Args:
            base_url: 基础URL
            remote_path: 远程文件所在目录
            file_list: 要下载的文件名列表
            local_dir: 本地保存目录

        Returns:
            Dict[str, bool]: 文件下载结果字典，键为文件名，值为是否下载成功
        """
        logger.info(f'开始下载 {len(file_list)} 个文件，最大并发数: {self.max_concurrency}')

        # 初始化全局进度条
        self.total_files = len(file_list)
        self.completed_files = 0

        # 创建全局进度条，位置在所有文件进度条之后
        position = min(self.max_concurrency, len(file_list))
        self.global_pbar = tqdm(
            desc='总进度', total=self.total_files, unit='个文件', position=position, leave=True
        )

        tasks = []
        for i, filename in enumerate(file_list):
            url = f'{base_url.strip("/")}/{remote_path.strip("/")}/{filename}'
            local_path = Path(local_dir) / filename

            position = i % self.max_concurrency

            task = asyncio.create_task(self.download_file(url, local_path, position=position))
            tasks.append((filename, task))

        results = {}
        for filename, task in tasks:
            try:
                success = await task
                results[filename] = success
                if not success:
                    self.failed_files.add(filename)
            except Exception as e:
                logger.error(f'下载文件 {filename} 时发生异常: {str(e)}')
                results[filename] = False
                self.failed_files.add(filename)

        # 关闭全局进度条
        if self.global_pbar:
            self.global_pbar.close()
            self.global_pbar = None

        success_count = sum(1 for success in results.values() if success)
        logger.info(f'下载完成: 成功 {success_count}/{len(file_list)} 个文件')

        self.download_results.update(results)
        return results

    async def retry_failed_files(
        self, base_url: str, remote_path: str, local_dir: str, max_retries: int = 3
    ) -> dict[str, bool]:
        """
        重试下载失败的文件

        Args:
            base_url: 基础URL
            remote_path: 远程文件所在目录
            local_dir: 本地保存目录
            max_retries: 最大重试次数

        Returns:
            Dict[str, bool]: 重试结果字典
        """
        retry_results = {}

        for retry in range(max_retries):
            if not self.failed_files:
                break

            logger.info(f'第 {retry + 1} 次重试下载失败的文件，共 {len(self.failed_files)} 个')

            current_failed = list(self.failed_files)
            self.failed_files.clear()

            results = await self.download_files(base_url, remote_path, current_failed, local_dir)
            retry_results.update(results)

            if not self.failed_files:
                logger.info('所有文件重试下载成功')
                break

            logger.info(f'第 {retry + 1} 次重试后仍有 {len(self.failed_files)} 个文件下载失败')

        return retry_results

    async def close(self):
        """关闭HTTP会话"""
        if self.session and not self.session.closed:
            await self.session.close()

        if self.global_pbar:
            self.global_pbar.close()
            self.global_pbar = None


async def download_http_files(
    base_url: str,
    remote_path: str,
    file_list: list[str],
    local_dir: str,
    max_concurrency: int = 5,
    retry_count: int = 3,
    timeout: int = 30,
) -> Optional[dict[str, bool]]:
    """
    便捷函数：下载HTTP文件

    Args:
        base_url: 基础URL
        remote_path: 远程文件所在目录
        file_list: 要下载的文件名列表
        local_dir: 本地保存目录
        max_concurrency: 最大并发下载数
        retry_count: 失败重试次数
        timeout: 超时时间（秒）

    Returns:
        Dict[str, bool]: 文件下载结果字典
    """
    downloader = HTTPDownloader(max_concurrency=max_concurrency, timeout=timeout)

    try:
        results = await downloader.download_files(base_url, remote_path, file_list, local_dir)

        if downloader.failed_files and retry_count > 0:
            retry_results = await downloader.retry_failed_files(
                base_url, remote_path, local_dir, max_retries=retry_count
            )
            results.update(retry_results)
    finally:
        await downloader.close()

    return results


if __name__ == '__main__':
    with FTPClient(host='ftp.ncbi.nlm.nih.gov', username='anonymous', password='') as ftp:
        files = ftp.list_files('/pubmed/baseline')
        logger.debug(files[0])

    async def download_example():
        await download_http_files(
            base_url='https://ftp.ncbi.nlm.nih.gov',
            remote_path='/pubmed/baseline',
            file_list=files[:10],
            local_dir='./downloads',
            max_concurrency=3,
        )

    asyncio.run(download_example())
