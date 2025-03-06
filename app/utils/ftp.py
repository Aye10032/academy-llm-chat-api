from ftplib import FTP
from typing import List, Optional, Tuple
from loguru import logger


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

    def list_files(self, remote_path: str = '.') -> List[Tuple[str, bool]]:
        """获取指定路径下的文件和目录列表

        Args:
            remote_path: 远程服务器上的路径，默认为当前目录

        Returns:
            List[Tuple[str, bool]]: 返回(文件名, 是否为目录)的列表
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
                    if name not in ('.', '..'):
                        file_list.append((name, is_dir))

            self.ftp.retrlines('LIST', process_line)
            file_list.sort(key=lambda x: (not x[1], x[0]))

            logger.info(f'成功获取目录 {remote_path} 下的文件列表，共 {len(file_list)} 个项目')
            return file_list

        except Exception as e:
            logger.error(f'获取文件列表失败: {str(e)}')
            return []


if __name__ == '__main__':
    with FTPClient(host='ftp.ncbi.nlm.nih.gov', username='anonymous', password='') as ftp:
        files = ftp.list_files('/pubmed/baseline')
        print(files[0])
