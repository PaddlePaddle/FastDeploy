#!/bin/env python3
import json
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BaselineManager:
    def __init__(self, base_dir="/ModelData"):
        self.base_dir = base_dir

    def _get_path(self, name: str):
        branch = os.getenv("TEST_BRANCH", "default")
        if os.getenv("STATIC_C8") == "1":
            c8_mode = "_static_c8"
        elif os.getenv("DYNAMIC_C8") == "1":
            c8_mode = "_dynamic_c8"
        else:
            c8_mode = ""
        return f"{self.base_dir}/{name}_{branch}{c8_mode}.txt"

    def load(self, name: str):
        path = self._get_path(name)
        logger.info(f"读取 baseline: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            return json.loads(content)
        except Exception:
            return content  # 不是 JSON，就当字符串返回

    def save(self, name: str, content):
        path = self._get_path(name)
        logger.info(f"写入 baseline: {path}")

        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False, indent=2)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
