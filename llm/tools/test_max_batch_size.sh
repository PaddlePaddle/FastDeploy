import fastdeploy_llm as fdlm
import sys
model_dir = sys.argv[1]
batch = int(sys.argv[2])
max_seq_len = int(sys.argv[3])
max_dec_len = int(sys.argv[4])
disable_dy_batch = 1
config = fdlm.Config(model_dir)
config.max_batch_size = batch
config.max_dec_len = max_dec_len
config.max_seq_len = max_seq_len
config.stop_threshold = 2
config.disable_dynamic_batching = disable_dy_batch
config.max_queue_num = 5120

model = fdlm.ServingModel(config)

for i in range(1000):
    task = fdlm.Task()
    task.text = "This is only for test"
    task.token_ids = [3] * max_seq_len
    task.max_dec_len = max_dec_len
    task.min_dec_len = max_dec_len
    task.penalty_score = 1.0
    task.temperature = 1.0
    task.topp = 0.0
    task.frequency_score = 0.0
    task.presence_score = 0.0
    task.task_id = i
    model.add_request(task)

model.start()
# 队列处理完成后即会自动退出
model.stop()
