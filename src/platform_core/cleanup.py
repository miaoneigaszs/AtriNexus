"""
清理工具模块
用于清理 Python 缓存文件和临时文件
"""

import os
import shutil
import logging
from typing import Tuple

logger = logging.getLogger('main')


def cleanup_pycache(root_dir: str = None, verbose: bool = False) -> Tuple[int, int]:
    """
    清理 Python 缓存文件
    
    清理内容:
    1. __pycache__ 目录
    2. .pyc 文件
    3. .pyo 文件
    
    Args:
        root_dir: 清理的根目录,默认为项目根目录
        verbose: 是否输出详细日志
    
    Returns:
        Tuple[int, int]: (清理的目录数, 清理的文件数)
    
    Example:
        >>> dirs, files = cleanup_pycache()
        >>> print(f"清理了 {dirs} 个目录, {files} 个文件")
    """
    if root_dir is None:
        # 默认使用项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    cleaned_dirs = 0
    cleaned_files = 0
    
    # 需要排除的目录
    exclude_dirs = {'.git', '.venv', 'venv', 'node_modules', '.idea', '.vscode', '__pycache__'}
    
    try:
        if verbose:
            logger.info(f"开始清理缓存文件: {root_dir}")
        
        for root, dirs, files in os.walk(root_dir, topdown=True):
            # 排除特定目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            # 清理 __pycache__ 目录
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                try:
                    shutil.rmtree(pycache_path)
                    cleaned_dirs += 1
                    if verbose:
                        logger.debug(f"已删除: {pycache_path}")
                except Exception as e:
                    if verbose:
                        logger.warning(f"删除失败 {pycache_path}: {e}")
            
            # 清理 .pyc 和 .pyo 文件
            for file in files:
                if file.endswith(('.pyc', '.pyo')):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        cleaned_files += 1
                        if verbose:
                            logger.debug(f"已删除: {file_path}")
                    except Exception as e:
                        if verbose:
                            logger.warning(f"删除失败 {file_path}: {e}")
        
        if verbose and (cleaned_dirs > 0 or cleaned_files > 0):
            logger.info(f"清理完成: {cleaned_dirs} 个目录, {cleaned_files} 个文件")
        
    except Exception as e:
        logger.warning(f"清理缓存时出错: {e}")
    
    return cleaned_dirs, cleaned_files


def cleanup_temp_files(root_dir: str = None, verbose: bool = False) -> int:
    """
    清理临时文件
    
    清理内容:
    1. .tmp 文件
    2. .bak 文件
    3. .swp 文件 (Vim 临时文件)
    
    Args:
        root_dir: 清理的根目录,默认为项目根目录
        verbose: 是否输出详细日志
    
    Returns:
        int: 清理的文件数
    """
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    cleaned_files = 0
    temp_extensions = {'.tmp', '.bak', '.swp'}
    
    # 需要排除的目录
    exclude_dirs = {'.git', '.venv', 'venv', 'node_modules', '.idea', '.vscode'}
    
    try:
        if verbose:
            logger.info(f"开始清理临时文件: {root_dir}")
        
        for root, dirs, files in os.walk(root_dir, topdown=True):
            # 排除特定目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            # 清理临时文件
            for file in files:
                if any(file.endswith(ext) for ext in temp_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        cleaned_files += 1
                        if verbose:
                            logger.debug(f"已删除: {file_path}")
                    except Exception as e:
                        if verbose:
                            logger.warning(f"删除失败 {file_path}: {e}")
        
        if verbose and cleaned_files > 0:
            logger.info(f"清理临时文件完成: {cleaned_files} 个文件")
        
    except Exception as e:
        logger.warning(f"清理临时文件时出错: {e}")
    
    return cleaned_files


def cleanup_logs(log_dir: str = None, max_age_days: int = 30, verbose: bool = False) -> int:
    """
    清理旧日志文件
    
    清理超过指定天数的日志文件
    
    Args:
        log_dir: 日志目录,默认为项目 logs 目录
        max_age_days: 最大保留天数,默认 30 天
        verbose: 是否输出详细日志
    
    Returns:
        int: 清理的文件数
    """
    if log_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(root_dir, 'logs')
    
    if not os.path.exists(log_dir):
        return 0
    
    cleaned_files = 0
    current_time = os.path.getmtime  # 使用文件的修改时间
    
    import time
    cutoff_time = time.time() - (max_age_days * 86400)  # 转换为秒
    
    try:
        if verbose:
            logger.info(f"开始清理旧日志: {log_dir} (保留 {max_age_days} 天)")
        
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if current_time(file_path) < cutoff_time:
                        os.remove(file_path)
                        cleaned_files += 1
                        if verbose:
                            logger.debug(f"已删除: {file_path}")
                except Exception as e:
                    if verbose:
                        logger.warning(f"删除失败 {file_path}: {e}")
        
        if verbose and cleaned_files > 0:
            logger.info(f"清理旧日志完成: {cleaned_files} 个文件")
        
    except Exception as e:
        logger.warning(f"清理旧日志时出错: {e}")
    
    return cleaned_files


def get_cache_size(root_dir: str = None) -> Tuple[int, int, int]:
    """
    获取缓存文件统计信息
    
    Args:
        root_dir: 统计的根目录,默认为项目根目录
    
    Returns:
        Tuple[int, int, int]: (目录数, 文件数, 总大小(字节))
    """
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    total_dirs = 0
    total_files = 0
    total_size = 0
    
    exclude_dirs = {'.git', '.venv', 'venv', 'node_modules', '.idea', '.vscode'}
    
    try:
        for root, dirs, files in os.walk(root_dir, topdown=True):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            # 统计 __pycache__ 目录
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                for sub_root, sub_dirs, sub_files in os.walk(pycache_path):
                    total_files += len(sub_files)
                    for file in sub_files:
                        try:
                            total_size += os.path.getsize(os.path.join(sub_root, file))
                        except:
                            pass
                total_dirs += 1
            
            # 统计 .pyc 和 .pyo 文件
            for file in files:
                if file.endswith(('.pyc', '.pyo')):
                    try:
                        total_size += os.path.getsize(os.path.join(root, file))
                        total_files += 1
                    except:
                        pass
        
    except Exception as e:
        logger.warning(f"统计缓存时出错: {e}")
    
    return total_dirs, total_files, total_size


def format_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
    
    Returns:
        str: 格式化后的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


if __name__ == '__main__':
    """测试清理功能"""
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("=== Python 缓存清理工具 ===\n")
    
    # 显示缓存统计
    dirs, files, size = get_cache_size()
    print(f"缓存统计:")
    print(f"  __pycache__ 目录: {dirs} 个")
    print(f"  缓存文件: {files} 个")
    print(f"  总大小: {format_size(size)}\n")
    
    # 确认清理
    if len(sys.argv) > 1 and sys.argv[1] == '-y':
        confirm = True
    else:
        confirm = input("是否清理缓存? (y/n): ").lower().strip() == 'y'
    
    if confirm:
        print("\n开始清理...")
        cleaned_dirs, cleaned_files = cleanup_pycache(verbose=True)
        print(f"\n清理完成:")
        print(f"  已删除目录: {cleaned_dirs} 个")
        print(f"  已删除文件: {cleaned_files} 个")
    else:
        print("已取消清理")
