"""eplb utilities"""

import json
import os
import time
from enum import Enum


class RedundantExpertWorkload:
    """Redundant Expert Workload"""

    def __init__(self, redundant_expert_meta_dir="/tmp/redundant_expert_meta"):
        self.update_timestamp = time.time()
        self.tokens_per_expert_stats_list = None
        self.ep_rank_to_expert_id_list = None
        self.expert_id_to_ep_rank_array = None
        self.expert_in_rank_num_list = None
        self.cost_milliseconds = 0
        self.meta_file_name = f"{redundant_expert_meta_dir}/rearrange-experts.json"
        if not os.path.exists(redundant_expert_meta_dir):
            os.makedirs(redundant_expert_meta_dir, exist_ok=True)

    def __json__(self):
        return self.__dict__

    def dump(self):
        """Dump the object to a JSON file."""
        begin = time.time()
        try:
            with open(self.meta_file_name, "w") as fout:
                json.dump(self.__dict__, fout)
        except Exception as e:
            return f"redundant_expert: dump expert workload failed, {e}"
        cost_time = int((time.time() - begin) * 1000 * 1000)
        return f"redundant_expert: dump expert workload result in {cost_time} us"

    def load(self):
        """Load the object from a JSON file."""
        if not os.path.exists(self.meta_file_name):
            return {}, f"redundant_expert: file {self.meta_file_name} is not exists"
        try:
            with open(self.meta_file_name, "r") as fin:
                meta = json.load(fin)
                self.__dict__.update(meta)
                return self.__json__(), "ok"
        except Exception as e:
            return {}, f"redundant_expert: load file {self.meta_file_name} failed, {e}"


class RearrangeExpertState(Enum):
    """RearrangeExpertState"""

    free = 0
    doing = 1
    load_succ = 2  # load weight from disk success
    done = 3


if __name__ == "__main__":
    print(RedundantExpertWorkload("/tmp").load())
