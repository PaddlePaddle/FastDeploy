import fastdeploy_llm as fdlm
import sys
model_dir = sys.argv[1]
data_path = sys.argv[2]
batch = int(sys.argv[3])
disable_dy_batch = int(sys.argv[4])
config = fdlm.Config(model_dir)
config.max_batch_size = batch
config.max_dec_len = 1024
config.max_seq_len = 1024
config.stop_threshold = 2
config.disable_dynamic_batching = disable_dy_batch
config.max_queue_num = 512
config.is_ptuning = int(sys.argv[5]) # enable ptuning
is_precache = int(sys.argv[6])  #enable precache
res_file = sys.argv[7]
if is_precache:
    config.model_prompt_dir_path = sys.argv[8]# 'prompt_embedding'
inputs = list()
with open(data_path, "r") as f:
    for line in f:
        data = eval(line.strip())
        prompt = data["src"]
        inputs.append((prompt, data))
 
model = fdlm.ServingModel(config)
def call_back(call_back_task, token_tuple, index, is_last_token, sender=None):
    with open("{}/{}".format(res_file,call_back_task.task_id), "a+") as f:
        f.write("{}\n".format(token_tuple))
 
for i, ipt in enumerate(inputs):
    task = fdlm.Task()
    task.text = ipt[0]
    if config.is_ptuning:
        task.max_dec_len = 1024-128
    else:
        task.max_dec_len = 1024
    task.min_dec_len = 1
    task.penalty_score = 1.0
    task.temperature = 1.0
    task.topp = 0.0
    task.frequency_score = 0.0
    task.eos_token_id = 2
    task.presence_score = 0.0
    task.task_id = i
    task.call_back_func = call_back
    if is_precache:
        task.model_id = 'test' #'test'
    else:
        task.model_id = None
    model.add_request(task)
 
model.start()
# 队列处理完成后即会自动退出
model.stop()
