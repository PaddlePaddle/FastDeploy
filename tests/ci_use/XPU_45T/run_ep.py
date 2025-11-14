import os

import psutil
from paddleformers.trainer import strtobool

from fastdeploy import LLM, SamplingParams


def test_fd_ep():
    """ """
    msg1 = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "北京天安门广场在哪里?"},
    ]
    messages = [msg1]
    print(f"[INFO] messages: {messages}")

    # 采样参数
    sampling_params = SamplingParams(top_p=0, max_tokens=500)

    # 模型路径与设备配置
    model_root = os.getenv("MODEL_PATH", "/home")
    model = f"{model_root}/ERNIE-4.5-300B-A47B-Paddle"
    xpu_visible_devices = os.getenv("XPU_VISIBLE_DEVICES", "0")
    xpu_device_num = len(xpu_visible_devices.split(","))

    enable_expert_parallel = strtobool(os.getenv("enable_expert_parallel", "1"))
    enable_tensor_parallel = strtobool(os.getenv("enable_tensor_parallel", "0"))
    disable_sequence_parallel_moe = strtobool(os.getenv("disable_sequence_parallel_moe", "0"))
    print(f"enable_expert_parallel: {enable_expert_parallel}, enable_tensor_parallel: {enable_tensor_parallel}")
    if enable_expert_parallel:
        if enable_tensor_parallel:
            tensor_parallel_size = xpu_device_num
            data_parallel_size = 1
        else:
            tensor_parallel_size = 1
            data_parallel_size = xpu_device_num
    else:
        tensor_parallel_size = xpu_device_num
        data_parallel_size = 1
    xpu_id = int(os.getenv("XPU_ID", "0"))
    base_port = 8023 + xpu_id * 100
    engine_worker_queue_port = [str(base_port + i * 10) for i in range(data_parallel_size)]
    engine_worker_queue_port = ",".join(engine_worker_queue_port)

    llm = LLM(
        model=model,
        enable_expert_parallel=enable_expert_parallel,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        disable_sequence_parallel_moe=disable_sequence_parallel_moe,
        max_model_len=8192,
        quantization="wint4",
        engine_worker_queue_port=engine_worker_queue_port,
        max_num_seqs=8,
        load_choices="default",
    )

    try:
        outputs = llm.chat(messages, sampling_params)
        assert outputs, "❌ LLM 推理返回空结果。"

        for idx, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = getattr(output.outputs, "text", "").strip()

            print(f"{'-'*100}")
            print(f"[PROMPT {idx}] {prompt}")
            print(f"{'-'*100}")
            print(f"[GENERATED TEXT] {generated_text}")
            print(f"{'-'*100}")

            # 核心断言：输出不能为空
            assert generated_text, f"❌ 推理结果为空 (index={idx})"

    finally:
        # 无论是否报错都清理子进程
        current_process = psutil.Process(os.getpid())
        for child in current_process.children(recursive=True):
            try:
                child.kill()
                print(f"[CLEANUP] 已杀死子进程 {child.pid}")
            except Exception as e:
                print(f"[WARN] 无法杀死子进程 {child.pid}: {e}")
        print("✅ 已清理所有 FastDeploy 子进程。")


if __name__ == "__main__":
    test_fd_ep()
