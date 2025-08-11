"""
自定义日志格式化器模块

该模块定义了 ColoredFormatter 类，用于在控制台输出带颜色的日志信息，
便于开发者在终端中快速识别不同级别的日志。
"""

import logging


class ColoredFormatter(logging.Formatter):
    """
    自定义日志格式器，用于控制台输出带颜色的日志。

    支持的颜色：
        - WARNING: 黄色
        - ERROR: 红色
        - CRITICAL: 红色
        - 其他等级: 默认终端颜色
    """

    COLOR_CODES = {
        logging.WARNING: 33,  # 黄色
        logging.ERROR: 31,  # 红色
        logging.CRITICAL: 31,  # 红色
    }

    def format(self, record):
        """
        格式化日志记录，并根据日志等级添加 ANSI 颜色前缀和后缀。

        Args:
            record (LogRecord): 日志记录对象。

        Returns:
            str: 带有颜色的日志消息字符串。
        """
        color_code = self.COLOR_CODES.get(record.levelno, 0)
        prefix = f"\033[{color_code}m"
        suffix = "\033[0m"
        message = super().format(record)
        if color_code:
            message = f"{prefix}{message}{suffix}"
        return message
