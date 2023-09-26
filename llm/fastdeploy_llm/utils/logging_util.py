# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging
import threading
import time
from typing import (Any, Generator, Optional, Union)
from logging.handlers import TimedRotatingFileHandler

import colorlog

__all__ = ['logger']

_LOG_CONFIG = {
    'DEBUG': {
        'color': 'purple'
    },
    'INFO': {
        'color': 'green'
    },
    'WARNING': {
        'color': 'yellow'
    },
    'ERROR': {
        'color': 'red'
    },
    'CRITICAL': {
        'color': 'bold_red'
    },
}


class Logger(object):
    _DEFAULT_NAME: str = 'FastDeploy'

    def __init__(self,
                 name: Optional[str]=None,
                 log_file=None,
                 time_rotation=7,
                 level=logging.INFO) -> None:
        """Initialize the instance based on a given name.

        Args:
            name: Logger name.
        """
        super().__init__()
        if name is None:
            name = self._DEFAULT_NAME
        self.logger = logging.getLogger(name)

        self.format = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s - %(message)s",
            log_colors={
                key: conf['color']
                for key, conf in _LOG_CONFIG.items()
            }, )

        if log_file is not None:
            self.handler = TimedRotatingFileHandler(
                log_file,
                when="midnight",
                backupCount=time_rotation,
                encoding="utf-8")
        else:
            self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logger.setLevel(level)
        self.logger.propagate = False
        self._is_enabled = True

    def __call__(self,
                 log_level: int,
                 msg: object,
                 *args: object,
                 **kwargs: Any) -> None:
        if not self.is_enabled:
            return

        self.logger.log(log_level, msg, *args, **kwargs)

    def debug(self, msg: object, *args: object, **kwargs: Any) -> None:
        return self(logging.getLevelName('DEBUG'), msg, *args, **kwargs)

    def info(self, msg: object, *args: object, **kwargs: Any) -> None:
        return self(logging.getLevelName('INFO'), msg, *args, **kwargs)

    def warning(self, msg: object, *args: object, **kwargs: Any) -> None:
        return self(logging.getLevelName('WARNING'), msg, *args, **kwargs)

    def error(self, msg: object, *args: object, **kwargs: Any) -> None:
        return self(logging.getLevelName('ERROR'), msg, *args, **kwargs)

    def critical(self, msg: object, *args: object, **kwargs: Any) -> None:
        return self(logging.getLevelName('CRITICAL'), msg, *args, **kwargs)

    def disable(self) -> None:
        self._is_enabled = False

    def enable(self) -> None:
        self._is_enabled = True

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    def set_level(self, log_level: Union[int, str]) -> None:
        self.logger.setLevel(log_level)

    @contextlib.contextmanager
    def processing(self, msg: str,
                   interval: float=0.1) -> Generator[None, None, None]:
        """Display a message with spinners.

        Args:
            msg: Message to display.
            interval: Spinning interval.
        """
        end = False

        def _printer() -> None:
            index = 0
            flags = ['\\', '|', '/', '-']
            while not end:
                flag = flags[index % len(flags)]
                with self.use_terminator('\r'):
                    self.info(f"{msg}: {flag}")
                time.sleep(interval)
                index += 1

        t = threading.Thread(target=_printer)
        t.start()
        yield
        end = True

    @contextlib.contextmanager
    def use_terminator(self, terminator: str) -> Generator[None, None, None]:
        old_terminator = self.handler.terminator
        self.handler.terminator = terminator
        yield
        self.handler.terminator = old_terminator


logger = Logger()
