import os
import multiprocessing
import json

class GlobalConfig():
    """ global config """

    def __init__(self):
        """init
        Args:
            None
        Returns:
            None
        """
        # Redis 
        self.redis_host = os.getenv('REDIS_HOST', default="localhost")
        self.redis_port = int(os.getenv('REDIS_PORT', default="6379"))
        self.redis_db = int(os.getenv('REDIS_DB', default="0"))
        self.redis_username = os.getenv('REDIS_USERNAME', default=None)
        self.redis_password = os.getenv('REDIS_PASSWORD', default=None)

        # Response
        self.resonpse_timeout = int(os.getenv('RESPONSE_TIMEOUT', default="120"))

        # Server
        self.num_process = int(os.getenv('NUM_PROCESS', default=multiprocessing.cpu_count()))

        # Logger
        self.log_dir = os.getenv('IC_LOG_DIR', default='ic_logs')
        