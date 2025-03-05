import asyncio
import fnmatch
import os
from urllib.parse import urlparse
from typing import Optional, Union, Any
import aioftp
from loguru import logger
from tqdm.asyncio import tqdm, tqdm_asyncio

from app.utils.md5 import verify_md5


class AsyncFTPDownloader:
    """
    异步FTP下载器类，用于从FTP服务器下载文件和目录
    """

    def __init__(
        self,
        host: str,
        port: int = 21,
        user: str = 'anonymous',
        password: str = '',
        timeout: int = 30,
        encoding: str = 'utf-8',
        binary_mode: bool = True,
        socks: Optional[str] = None,
    ):
        """
        初始化FTP下载器

        Args:
            host: FTP服务器地址 (主机名如'example.com')
            port: FTP服务器端口，默认为21
            user: FTP用户名，默认为匿名用户
            password: FTP密码，默认为空
            timeout: 连接超时时间（秒），默认为30秒
            encoding: 文件名编码，默认为utf-8
            binary_mode: 是否使用二进制模式传输，默认为True
            socks: 是否使用代理，传入完整的代理地址 (http://localhost:7890 等)
        """
        self.host = host.strip()
        self.port = port
        self.user = user
        self.password = password
        self.timeout = timeout
        self.encoding = encoding
        self.binary_mode = binary_mode
        self.socks = socks

        self.client: Optional[aioftp.Client] = None

    async def connect(self) -> None:
        try:
            if self.socks:
                parsed_socks = urlparse(self.socks)
                socks_host = parsed_socks.hostname
                socks_port = parsed_socks.port

                if not socks_host or not socks_port:
                    raise ValueError(
                        f'无效的SOCKS代理格式: {self.socks}，正确格式应为 socks5://host:port'
                    )

                logger.info(f'使用SOCKS代理: {socks_host}:{socks_port}')

                self.client = aioftp.Client(
                    encoding=self.encoding,
                    socket_timeout=self.timeout,
                    connection_timeout=self.timeout,
                    path_timeout=self.timeout,
                    socks_host=socks_host,
                    socks_port=socks_port,
                    socks_version=5 if parsed_socks.scheme == 'socks5' else 4,
                )
            else:
                self.client = aioftp.Client(
                    encoding=self.encoding,
                    socket_timeout=self.timeout,
                    connection_timeout=self.timeout,
                    path_timeout=self.timeout,
                )
            logger.info(f'正在连接到FTP服务器: {self.host}:{self.port}')
            await self.client.connect(self.host, self.port)
            await self.client.login(self.user, self.password)

            try:
                if self.binary_mode:
                    await self.client.command('TYPE', 'I')
                else:
                    await self.client.command('TYPE', 'A')
            except Exception as e:
                logger.warning(f'设置传输模式失败: {str(e)}，将使用默认模式')

            logger.info(f'已连接到FTP服务器: {self.host}:{self.port}')
        except Exception as e:
            logger.exception(f'连接FTP服务器失败: {str(e)}')
            raise

    async def disconnect(self) -> None:
        if self.client and self.client.connect:
            await self.client.quit()
            logger.info(f'已断开与FTP服务器的连接: {self.host}:{self.port}')
            self.client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def list_directory(self, remote_path: str = '.') -> list[dict[str, Any]]:
        """列出远程目录中的文件和子目录

        Args:
            remote_path: 远程目录路径，默认为当前目录

        Returns:
            文件和目录信息的列表
        """
        if not self.client:
            await self.connect()

        try:
            result = []
            async for path, info in self.client.list(remote_path, recursive=False):
                item = {
                    'name': path.name,
                    'path': str(path),
                    'is_dir': info['type'] == 'dir',
                    'size': info.get('size', 0),
                    'modify_time': info.get('modify', None),
                }
                result.append(item)
            return result
        except Exception as e:
            logger.exception(f'列出目录失败 {remote_path}: {str(e)}')
            raise

    async def download_file(
        self, remote_file: dict[str, str], local_path: str, overwrite: bool = False
    ) -> bool:
        """下载单个文件

        Args:
            remote_file: 远程文件路径
            local_path: 本地保存路径
            overwrite: 是否覆盖已存在的文件，默认为False

        Returns:
            下载是否成功
        """
        remote_name = remote_file['name']
        remote_path = remote_file['path']

        if os.path.exists(os.path.join(local_path, remote_name)) and not overwrite:
            if remote_name.endswith('.md5'):
                return True
            else:
                if os.path.exists(os.path.join(local_path, f'{remote_name}.md5')):
                    if verify_md5(
                        os.path.join(local_path, remote_name),
                        os.path.join(local_path, f'{remote_name}.md5'),
                    ):
                        return True

        if not self.client:
            await self.connect()

        try:
            await self.client.download(remote_path, local_path)
            return True
        except Exception as e:
            logger.exception(f'下载文件失败 {remote_path}: {str(e)}')
            return False

    @staticmethod
    async def download(
        host: str,
        remote_path: str,
        local_path: str,
        port: int = 21,
        user: str = 'anonymous',
        password: str = '',
        binary_mode: bool = True,
        is_file: bool = True,
        **kwargs,
    ) -> Union[bool, dict[str, int]]:
        """静态便捷方法，用于快速下载文件或目录

        Args:
            host: FTP服务器地址 (可以是完整URL如'ftp://example.com'或主机名如'example.com')
            remote_path: 远程文件或目录路径
            local_path: 本地保存路径
            port: FTP服务器端口，默认为21
            user: FTP用户名，默认为匿名用户
            password: FTP密码，默认为空
            binary_mode: 是否使用二进制模式传输，默认为True
            is_file: 是否为文件下载，默认为True
            **kwargs: 其他参数传递给download_file或download_directory方法

        Returns:
            下载结果
        """
        actual_host = host.lstrip('ftp:').strip('/')

        async with AsyncFTPDownloader(
            actual_host, port, user, password, binary_mode=binary_mode
        ) as downloader:
            if is_file:
                file_name = remote_path.split('/')[-1]
                return await downloader.download_file(
                    {'path': remote_path, 'name': file_name}, local_path, **kwargs
                )


async def concurrent_download_files(
    host: str,
    remote_dir: str,
    local_dir: str,
    port: int = 21,
    user: str = 'anonymous',
    password: str = '',
    max_concurrency: int = 5,
    overwrite: bool = False,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    binary_mode: bool = True,
    timeout: int = 30,
) -> dict[str, int]:
    """先获取文件列表，然后并发下载文件

    Args:
        host: FTP服务器地址
        remote_dir: 远程目录路径
        local_dir: 本地保存目录
        port: FTP服务器端口
        user: FTP用户名
        password: FTP密码
        max_concurrency: 最大并发数
        overwrite: 是否覆盖已存在文件
        include_pattern: 包含的文件模式（glob格式）
        exclude_pattern: 排除的文件模式（glob格式）
        binary_mode: 是否使用二进制模式
        timeout: 连接超时时间

    Returns:
        包含成功和失败计数的字典
    """
    os.makedirs(local_dir, exist_ok=True)

    # 统计结果
    stats = {'success': 0, 'failed': 0, 'skipped': 0}

    try:
        async with AsyncFTPDownloader(
            host, port, user, password, timeout=timeout, binary_mode=binary_mode
        ) as list_downloader:
            files = await list_downloader.list_directory(remote_dir)
            logger.info(f'共计 {len(files)} 个文件/目录')

            filtered_files = files
            if include_pattern:
                filtered_files = [
                    f for f in filtered_files if fnmatch.fnmatch(f['name'], include_pattern)
                ]
            if exclude_pattern:
                filtered_files = [
                    f for f in filtered_files if not fnmatch.fnmatch(f['name'], exclude_pattern)
                ]

            logger.info(f'过滤后剩余 {len(filtered_files)} 个文件/目录')

        logger.info(f'开始并发下载，最大并发数: {max_concurrency}')

        semaphore = asyncio.Semaphore(max_concurrency)

        async def download_single_file(file):
            """下载单个文件的任务"""
            async with semaphore:
                await AsyncFTPDownloader.download(
                    host,
                    file,
                    local_dir,
                    user=user,
                    password=password,
                    binary_mode=binary_mode,
                    is_file=True,
                    overwrite=overwrite
                )

        download_tasks = [download_single_file(file['path']) for file in filtered_files]

        results = await tqdm_asyncio.gather(
            *download_tasks, desc='FTP异步下载进度', total=len(download_tasks)
        )

        stats['success'] = results.count('success')
        stats['failed'] = results.count('failed')
        stats['skipped'] = results.count('skipped')

        logger.info(
            f'目录下载完成: {remote_dir} -> {local_dir}，'
            f'成功: {stats["success"]}，'
            f'失败: {stats["failed"]}，'
            f'跳过: {stats["skipped"]}'
        )
        return stats

    except Exception as e:
        logger.exception(f'下载目录失败 {remote_dir}: {str(e)}')
        raise
