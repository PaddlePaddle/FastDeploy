# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import json, copy, json, struct, pickle, glob, traceback, queue, threading, time
from multiprocessing import shared_memory
from collections import Counter, namedtuple
import numpy as np
import sys, os, codecs
import queue
import signal

from fastdeploy_llm.processor import DataProcessor
from fastdeploy_llm.utils.launch_infer import launch
from fastdeploy_llm.utils.utils import deserialize_from_file, get_files, remove_files
from fastdeploy_llm.config import Config
from fastdeploy_llm.task import Task, TaskStatus
from fastdeploy_llm.utils.logging_util import logger

from concurrent.futures import ThreadPoolExecutor

if sys.stdout.encoding is None:
    enc = os.environ['LANG'].split('.')[1]
    sys.stdout = codecs.getwriter(enc)(sys.stdout)


class Model:
    def _init_share_memory(self):
        flag_array = np.zeros([self.config.mp_num], dtype=np.int32)
        try:
            tmp = shared_memory.SharedMemory(
                create=False,
                size=flag_array.nbytes,
                name="shm_pd_infer_flag_begin")
            tmp.close()
            tmp.unlink()
        except:
            pass

        self.shm_flag_begin = shared_memory.SharedMemory(
            create=True,
            size=flag_array.nbytes,
            name="shm_pd_infer_flag_begin")
        self.flag_begin_array = np.ndarray(
            flag_array.shape,
            dtype=flag_array.dtype,
            buffer=self.shm_flag_begin.buf)
        self.flag_begin_array[:] = 0

        try:
            tmp = shared_memory.SharedMemory(
                create=False,
                size=flag_array.nbytes,
                name="shm_pd_infer_flag_end")
            tmp.close()
            tmp.unlink()
        except:
            pass
        self.shm_flag_end = shared_memory.SharedMemory(
            create=True, size=flag_array.nbytes, name="shm_pd_infer_flag_end")
        self.flag_end_array = np.ndarray(
            flag_array.shape,
            dtype=flag_array.dtype,
            buffer=self.shm_flag_end.buf)
        self.flag_end_array[:] = 0

        try:
            tmp = shared_memory.SharedMemory(
                create=False,
                size=flag_array.nbytes,
                name="shm_flag_infer_ready")
            tmp.close()
            tmp.unlink()
        except:
            pass
        self.shm_flag_ready = shared_memory.SharedMemory(
            create=True, size=flag_array.nbytes, name="shm_flag_infer_ready")
        self.flag_ready_array = np.ndarray(
            flag_array.shape,
            dtype=flag_array.dtype,
            buffer=self.shm_flag_ready.buf)
        self.flag_ready_array[:] = 0

        output_datas = dict()
        keys = [
            "finished_ids", "step_idx", "stop_flags", "seq_lens_decoder",
            "tgt_pos"
        ]
        for i in range(5):
            output_datas[keys[i]] = np.zeros(
                [self.config.max_batch_size + 1, 1], dtype=np.int64)
        if self.config.is_arch("chatglm"):
            output_datas["tgt_pos"] = np.zeros(
                [self.config.max_batch_size + 1, 2, 1], dtype=np.int64)
        output_datas_size = len(pickle.dumps(output_datas)) + 6
        self.shm_output_data = shared_memory.SharedMemory(
            create=True, size=output_datas_size, name="shm_infer_output_data")

    def _notify_engine(self):
        self.flag_begin_array[:] = 1
        self.flag_end_array[:] = 0

    def _is_engine_initialized(self):
        return np.sum(self.flag_ready_array) == self.config.mp_num

    def _is_engine_busy(self):
        if np.sum(self.flag_begin_array) > 0 and np.sum(
                self.flag_end_array) != self.config.mp_num:
            return True
        return False

    def _request_to_engine(self, inputs):
        input_dump_str = pickle.dumps(inputs)
        self.shm_input_data = shared_memory.SharedMemory(
            create=True, size=len(input_dump_str), name="shm_infer_input_data")
        self.shm_input_data.buf[:] = input_dump_str

    def _clear_request(self):
        self.shm_input_data.close()
        self.shm_input_data.unlink()

    def _get_engine_info(self):
        output_size = bytes(self.shm_output_data.buf[:4])
        output_size = int.from_bytes(output_size, 'big')
        info = pickle.loads(bytes(self.shm_output_data.buf[4:output_size + 4]))
        return info

    def kill_engine(self):
        if self.engine_proc is not None:
            os.killpg(self.engine_proc.pid, signal.SIGTERM)

    def _init_engine(self, configs):
        device_ids = ",".join(str(i) for i in range(self.config.mp_num))
        device_ids = os.getenv("CUDA_VISIBLE_DEVICES", device_ids)

        keyword_args = {
            'model_dir': configs.model_dir,
            'batch_size': configs.max_batch_size,
            'max_seq_len': configs.max_seq_len,
            'max_dec_len': configs.max_dec_len,
            'num_layers': configs.num_layers,
            'num_attention_heads': configs.num_attention_heads,
            'hidden_size': configs.hidden_size,
            'architecture': configs.architecture,
            'is_static_model': int(configs.is_static_model),
            "decode_strategy": configs.decode_strategy,
            # below configurations only take effect when is_ptuning is on
            "is_ptuning": configs.is_ptuning,
            "model_prompt_dir_path": configs.model_prompt_dir_path,
            "max_prefix_len": configs.max_prefix_len
        }
        self.engine_proc = launch(device_ids, **keyword_args)
        while not self._is_engine_initialized():
            time.sleep(1)
            ret = self.engine_proc.poll()
            if ret is not None:
                logger.error(
                    "The engine launch failed, check log/workerlog for more details."
                )
                raise Exception(
                    "The engine launch failed, check log/workerlog for more details."
                )
        logger.info("Paddle Inference Engine intialized successed!")

    def __init__(self, config):
        self.thread_executor = ThreadPoolExecutor(max_workers=1)
        self.config = config
        self.engine_proc = None
        self._init_share_memory()
        self._init_engine(self.config)
        self.data_processor = DataProcessor(self.config.model_dir)
        # stream_sender is a reserved value for serving frameworks such as triton
        # to send generated tokens streamly
        self.stream_sender = None

    def _pad_noencoder_batch_task(self, batch_tasks):
        new_task_counter = 0
        for i in range(batch_tasks.size()):
            if batch_tasks.tasks[i].status == TaskStatus.NEW:
                new_task_counter += 1

        if new_task_counter > 0:
            return False

        ignored_task = copy.deepcopy(batch_tasks.tasks[0])
        ignored_task.clear()
        ignored_task.call_back_func = None
        ignored_task.text = "me"
        ignored_task.max_dec_len = 2
        ignored_task.is_pad = True
        batch_tasks.update([ignored_task])
        logger.info(
            "There's no new task in batch_tasks, will insert a pad task to it.")
        return True

    def async_predict(self, batch_tasks, stop_num=None):
        if stop_num is None:  # 默认等待所有结果完成
            stop_num = batch_tasks.size()

        if batch_tasks.unfinished_size() == 0:
            raise Exception(
                "There's no unfinished tasks, please insert new tasks.")
        if batch_tasks.size() > self.config.max_batch_size:
            raise Exception(
                "The input tasks size {} is exceed the max batch size {}.".
                format(batch_tasks.size(), self.config.max_batch_size))
        if self._is_engine_busy():
            raise Exception("The inference engine is still running.")
        if stop_num <= 0:
            raise Exception("The stop_num should greater than 0.")

        if self._pad_noencoder_batch_task(batch_tasks):
            stop_num += 1
        inputs = self._prepare_inputs(batch_tasks.tasks, stop_num)
        self._request_to_engine(inputs)
        self._notify_engine()
        return self.thread_executor.submit(self._update_task_results,
                                           batch_tasks.tasks)

    def _update_task_results(self, tasks):
        step_index = 1
        while True:
            filepath = f"./real_time_save.temp_ids_rank_0_step_{step_index}"
            if os.path.exists(filepath):
                while True:
                    fin = open(filepath, "rb+")
                    try:
                        if fin.read(1) == b'1':
                            break
                    except:
                        fin.close()
                token_ids = deserialize_from_file(fin)
                fin.close()
                step_index += 1
                for b, token_id in enumerate(token_ids):
                    token_id = int(token_id)
                    if token_id == -1 or tasks[
                            b].status == TaskStatus.FINISHED:
                        continue
                    if token_id not in self.config.eos_token_id:
                        # TODO There's some differs between decode and batch decode
                        # will fix it latter
                        previous_token_ids = [
                            t[0] for t in tasks[b].result.completion_tokens
                        ]
                        decode_token_str, tasks[b].result.decode_prefix_offset, tasks[
                            b].result.decode_read_offset = self.data_processor.decode_token(
                                previous_token_ids + [token_id],
                                tasks[b].result.decode_prefix_offset,
                                tasks[b].result.decode_read_offset)
                        tasks[b].add_generated_token(
                            token_id,
                            decode_token_str,
                            False,
                            sender=self.stream_sender)
                        tasks[b].status = TaskStatus.DECODING
                    else:
                        decode_token_str = ""
                        tasks[b].add_generated_token(
                            token_id,
                            decode_token_str,
                            True,
                            sender=self.stream_sender)
                        tasks[b].status = TaskStatus.FINISHED
                    if step_index > tasks[b].max_dec_len:
                        if tasks[b].status != TaskStatus.FINISHED:
                            tasks[b].add_generated_token(
                                list(self.config.eos_token_id)[0],
                                "",
                                True,
                                sender=self.stream_sender)
                        tasks[b].status = TaskStatus.FINISHED
            else:
                if not self._is_engine_busy():
                    break
                ret = self.engine_proc.poll()
                if ret is not None:
                    logger.error(
                        "The inference engine is not alive, check log/workerlog for more details."
                    )
                    raise Exception(
                        "The inference engine is not alive, check log/workerlog for more details."
                    )

        remove_files(".", "real_time_save.temp_ids_rank_*")
        self._clear_request()
        info = self._get_engine_info()
        for i in range(len(tasks)):
            tasks[i].decode_status["finished_id"] = int(info["finished_ids"][
                i])
            if self.config.is_arch("chatglm"):
                tasks[i].decode_status["tgt_pos"] = info["tgt_pos"][i].flatten(
                ).tolist()
            else:
                tasks[i].decode_status["tgt_pos"] = int(info["tgt_pos"][i])

            tasks[i].decode_status["step_idx"] = int(info["step_idx"][i])
            tasks[i].decode_status["seq_lens_decoder"] = int(info[
                "seq_lens_decoder"][i])
            if info["stop_flags"][i] == 1:
                if tasks[i].status != TaskStatus.FINISHED:
                    logger.error(
                        "There's differ in FINISHED check for task_id={}, text={}".
                        format(tasks[i].task_id, tasks[i].text))
                    tasks[i].add_generated_token(
                        list(self.config.eos_token_id)[0],
                        "",
                        True,
                        sender=self.stream_sender)
                    tasks[i].status = TaskStatus.FINISHED
            else:
                tasks[i].status = TaskStatus.DECODING

    def _get_engine_info(self):
        output_size = bytes(self.shm_output_data.buf[:4])
        output_size = int.from_bytes(output_size, 'big')
        info = pickle.loads(bytes(self.shm_output_data.buf[4:output_size + 4]))
        return info

    def predict(self, tasks, stop_num=None):
        result = self.async_predict(tasks, stop_num)
        if result is None:
            return None
        result.result()  # 阻塞，直到任务完成

    def _prepare_inputs(self, tasks, stop_num):
        inputs = dict()
        texts = []
        for task in tasks:
            if task.status == TaskStatus.NEW:
                texts.append(task.text)
            else:
                texts.append("me")
                if hasattr(task, "token_ids"):
                    del task.token_ids
                if hasattr(task, "position_ids"):
                    del task.position_ids

        input_ids, lens, position_ids = self.data_processor.batch_encode_tasks(
            tasks, padding=True)

        for i in range(len(tasks)):
            tasks[i].prompt_token_nums = lens[i]
            tasks[i].prompt_token_ids = input_ids[i]
        inputs["input_ids"] = np.array(input_ids).astype("int64")
        inputs["top_p"] = np.array([task.topp for task in tasks]).reshape(
            -1, 1).astype("float32")
        inputs["temperature"] = np.array(
            [task.temperature for task in tasks]).reshape(-1,
                                                          1).astype("float32")
        inputs["penalty_score"] = np.array(
            [task.penalty_score for task in tasks]).astype('float32').reshape(
                -1, 1)
        inputs["presence_score"] = np.array(
            [task.presence_score for task in tasks]).astype('float32').reshape(
                -1, 1)
        inputs["frequency_score"] = np.array(
            [task.frequency_score
             for task in tasks]).astype('float32').reshape(-1, 1)
        inputs["min_length"] = np.array(
            [task.min_dec_len for task in tasks]).astype('int64').reshape(-1,
                                                                          1)
        # TODO Lite model exists different method
        # TODO Doesn't support eos_token id now
        inputs["eos_token_id"] = np.array(
            [list(self.config.eos_token_id)[0]
             for task in tasks]).astype('int64').reshape(-1, 1)
        inputs["max_length"] = np.array(
            [task.max_dec_len for task in tasks], dtype="int64").reshape(-1, 1)
        inputs["num_input_tokens"] = lens

        inputs = self._add_dynamic_batching_inputs(inputs, tasks, stop_num)
        return inputs

    def _add_dynamic_batching_inputs(self, inputs, tasks, stop_nums):
        bsz = len(tasks)
        tgt_pos = list()
        sequence_lengths_encoder = [0] * bsz
        sequence_lengths_decoder = [0] * bsz
        tgt_ids = [0] * bsz
        stop_flags = [0] * bsz
        step_idx = [0] * bsz
        dyinput_flags = [0] * bsz
        if self.config.is_ptuning:
            inputs["model_id"] = []
        for i in range(len(tasks)):
            if self.config.is_ptuning:
                inputs["model_id"].append(tasks[i].model_id)
            length = inputs["num_input_tokens"][i]
            if tasks[i].status == TaskStatus.DECODING:
                if self.config.is_arch("chatglm"):
                    tgt_pos += [
                        tasks[i].decode_status["seq_lens_decoder"] -
                        tasks[i].decode_status["step_idx"] + 1,
                        tasks[i].decode_status["step_idx"]
                    ]
                else:
                    tgt_pos.append(tasks[i].decode_status["tgt_pos"])
                sequence_lengths_encoder[i] = 0
                sequence_lengths_decoder[i] = tasks[i].decode_status[
                    "seq_lens_decoder"]
                inputs["input_ids"][i][0] = tasks[i].decode_status[
                    "finished_id"]
                tgt_ids[i] = tasks[i].decode_status["finished_id"]
                step_idx[i] = tasks[i].decode_status["step_idx"]
            elif tasks[i].status == TaskStatus.FINISHED:
                if self.config.is_arch("chatglm"):
                    tgt_pos.append(length)
                    tgt_pos.append(1)
                else:
                    tgt_pos.append(length - 1)
                sequence_lengths_encoder[i] = 0
                if self.config.is_ptuning:
                    sequence_lengths_decoder[i] = self.config.max_prefix_len
                else:
                    sequence_lengths_decoder[i] = 0
                tgt_ids[i] = inputs["input_ids"][i][length - 1]
                stop_flags[i] = 1
            elif tasks[i].status == TaskStatus.NEW:
                if self.config.is_arch("chatglm"):
                    tgt_pos += [length, 1]
                else:
                    tgt_pos.append(length - 1)
                sequence_lengths_encoder[i] = length

                if self.config.is_ptuning:
                    if not getattr(tasks[i], "is_pad", False):
                        sequence_lengths_decoder[
                            i] = length + self.config.max_prefix_len
                    else:
                        sequence_lengths_decoder[i] = 0
                        stop_flags[i] = 1
                else:
                    if not getattr(tasks[i], "is_pad", False):
                        sequence_lengths_decoder[i] = length
                    else:
                        sequence_lengths_decoder[i] = 0
                        stop_flags[i] = 1
                tgt_ids[i] = inputs["input_ids"][i][length - 1]
                dyinput_flags[i] = 1
        del inputs["num_input_tokens"]
        if self.config.is_arch("chatglm"):
            tgt_pos = np.array(tgt_pos).astype("int64").reshape(-1, 2, 1)
        else:
            tgt_pos = np.array(tgt_pos).astype("int64").reshape(-1, 1)
        sequence_lengths_encoder = np.array(sequence_lengths_encoder).astype(
            "int32").reshape(-1, 1)
        sequence_lengths_decoder = np.array(sequence_lengths_decoder).astype(
            "int32").reshape(-1, 1)
        tgt_ids = np.array(tgt_ids).astype("int64").reshape(-1, 1)
        stop_flags = np.array(stop_flags).astype("bool").reshape(-1, 1)
        step_idx = np.array(step_idx).astype("int64").reshape(-1, 1)

        finished_num = np.sum(stop_flags)
        if stop_nums < finished_num:
            stop_nums = finished_num + 1
        if stop_nums > bsz:
            stop_nums = bsz

        stop_nums = np.array(stop_nums).astype("int64")
        inputs.update({
            "tgt_pos": tgt_pos,
            "seq_len_encoder": sequence_lengths_encoder,
            "seq_len_decoder": sequence_lengths_decoder,
            "tgt_ids": tgt_ids,
            "stop_flags": stop_flags,
            "step_idx": step_idx,
            "stop_nums": np.array(stop_nums).reshape([-1]).astype("int64"),
            "dyinput_flags": dyinput_flags
        })
        return inputs
