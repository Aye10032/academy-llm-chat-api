import hashlib
from typing import Optional

from loguru import logger


def calculate_md5(file_path: str) -> Optional[str]:
    """计算文件的 MD5 值

    Args:
        file_path: 待校验的原始文件

    Returns:
        计算得到的MD5值
    """
    md5_hash = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except FileNotFoundError:
        return None


def read_md5_from_file(md5_file_path: str) -> Optional[str]:
    """从 .md5 文件读取 MD5 值

    Args:
        md5_file_path: MD5文件

    Returns:
        读取的MD5码
    """
    try:
        with open(md5_file_path, 'r') as f:
            line = f.readline().strip()
            md5_value = line.split(' ')[-1]
            return md5_value
    except FileNotFoundError:
        return None


def verify_md5(file_path: str, md5_file_path: str) -> bool:
    """校验文件 MD5

    Args:
        file_path: 待校验的原始文件
        md5_file_path: MD5文件

    Returns:
        校验结果
    """
    calculated_md5 = calculate_md5(file_path)
    expected_md5 = read_md5_from_file(md5_file_path)

    if calculated_md5 is None:
        logger.error(f'错误：文件 {file_path} 未找到。')
        return False

    if expected_md5 is None:
        logger.error(f'错误：MD5 文件 {md5_file_path} 未找到。')
        return False

    if calculated_md5.lower() == expected_md5.lower():
        return True
    else:
        logger.error('MD5 校验失败！')
        logger.error(f'  计算得到的 MD5: {calculated_md5}')
        logger.error(f'  期望的 MD5: {expected_md5}')
        return False

def main():
    file_path = 'downloaded/pubmed25n0001.xml.gz'
    md5_file_path = 'downloaded/pubmed25n0001.xml.gz.md5'

    verify_md5(file_path, md5_file_path)

if __name__ == '__main__':
    main()
