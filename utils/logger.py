import logging
import os
from datetime import datetime

def setup_logger(script_function=None):
    """
    设置集中式日志配置
    
    Args:
        script_function: 脚本函数名，用于生成日志文件名
    
    Returns:
        logger: 配置好的logger实例
    """
    # 创建logs目录
    log_dir = "experiments/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    if script_function is None:
        script_function = "main"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_function}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # 创建logger
    logger = logging.getLogger(script_function)
    logger.setLevel(logging.INFO)
    
    # 清除现有的handlers（避免重复）
    logger.handlers.clear()
    
    # 设置日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s'
    )
    
    # 添加StreamHandler（输出到终端）
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 添加FileHandler（输出到文件）
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Log file: {log_filepath}")
    
    return logger

# 创建默认logger实例
logger = setup_logger("default") 