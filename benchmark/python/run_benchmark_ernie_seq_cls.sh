# Download and decompress the ERNIE 3.0 Medium model finetuned on AFQMC
# wget https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-afqmc.tgz
# tar xvfz ernie-3.0-medium-zh-afqmc.tgz

# Download and decompress the quantization model of ERNIE 3.0 Medium model
# wget https://bj.bcebos.com/fastdeploy/models/ernie-3.0/ernie-3.0-medium-zh-afqmc-new-quant.tgz
# tar xvfz ernie-3.0-medium-zh-afqmc-new-quant.tgz

# PP-TRT
python benchmark_ernie_seq_cls.py --batch_size 40 --model_dir ernie-3.0-medium-zh-afqmc --backend pp-trt
python benchmark_ernie_seq_cls.py --batch_size 40 --model_dir ernie-3.0-medium-zh-afqmc --backend pp-trt --use_fp16 True
python benchmark_ernie_seq_cls.py --batch_size 40 --model_dir ernie-3.0-medium-zh-afqmc-new-quant --backend pp-trt

# TRT
python benchmark_ernie_seq_cls.py --batch_size 40 --model_dir ernie-3.0-medium-zh-afqmc --backend trt
python benchmark_ernie_seq_cls.py --batch_size 40 --model_dir ernie-3.0-medium-zh-afqmc --backend trt --use_fp16 True
python benchmark_ernie_seq_cls.py --batch_size 40 --model_dir ernie-3.0-medium-zh-afqmc-new-quant --backend trt --use_fp16 True

# CPU PP
python benchmark_ernie_seq_cls.py --batch_size 40 --cpu_num_threads 10 --model_dir ernie-3.0-medium-zh-afqmc --backend pp --device cpu
python benchmark_ernie_seq_cls.py --batch_size 40 --cpu_num_threads 10 --model_dir ernie-3.0-medium-zh-afqmc-new-quant --backend pp --device cpu

# CPU ORT
python benchmark_ernie_seq_cls.py --batch_size 40 --cpu_num_threads 10 --model_dir ernie-3.0-medium-zh-afqmc --backend ort --device cpu
python benchmark_ernie_seq_cls.py --batch_size 40 --cpu_num_threads 10 --model_dir ernie-3.0-medium-zh-afqmc-new-quant --backend ort --device cpu
