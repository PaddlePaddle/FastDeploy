#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ServeTest
"""
import logging
import os
from datetime import datetime

import pytz


class Logger(object):
    """
    日志记录配置的基础类。
    """

    SAVE_LEVELS = ["both", "file", "channel"]
    LOG_FORMAT = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"

    def __init__(self, loggername, save_level="both", log_path=None):
        """
        使用指定名称和保存级别初始化日志记录器。

        Args:
            loggername (str): 日志记录器的名称。
            save_level (str): 日志保存的级别。默认为"both"。file: 仅保存到文件，channel: 仅保存到控制台。
            log_path (str, optional): 日志文件保存路径。默认为None。
        """

        if save_level not in self.SAVE_LEVELS:
            raise ValueError(f"Invalid save level: {save_level}. Allowed values: {self.SAVE_LEVELS}")

        self.logger = logging.getLogger(loggername)
        self.logger.setLevel(logging.DEBUG)

        # 设置时区为东八区
        tz = pytz.timezone("Asia/Shanghai")

        # 自定义时间格式化器，指定时区为东八区
        class CSTFormatter(logging.Formatter):
            """
            自定义时间格式化器，指定时区为东八区
            """

            def converter(self, timestamp):
                """
                自定义时间转换函数，加上时区信息
                Args:
                    timestamp (int): 时间戳。
                Returns:
                    tuple: 格式化后的时间元组。
                """
                dt = datetime.utcfromtimestamp(timestamp)
                dt = pytz.utc.localize(dt).astimezone(tz)
                return dt.timetuple()

        formatter = CSTFormatter(self.LOG_FORMAT)
        log_name = None
        if save_level == "both" or save_level == "file":
            os.makedirs(log_path, exist_ok=True)
            log_filename = f"out_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
            log_name = os.path.join(log_path, log_filename)
            file_handler = logging.FileHandler(log_name, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        if save_level == "both" or save_level == "channel":
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if log_name is None:
            self.logger.info(
                f"Logger initialized. Log level: {save_level}. "
                f"Log path ({log_path}) is unused according to the level."
            )
        else:
            self.logger.info(f"Logger initialized. Log level: {save_level}. Log path: {log_name}")
            # Adjusting the timezone offset

    def get_logger(self):
        """
        Get the logger object
        """
        return self.logger


if __name__ == "__main__":
    # Test the logger
    logger = Logger("test_logger", save_level="channel").get_logger()
    logger.info("the is the beginning")
    logger.debug("the is the beginning")
    logger.warning("the is the beginning")
    logger.error("the is the beginning")
