# PP-TRT
python infer.py --batch_size 40 --model_dir ernie_float32 --backend pp-trt
python infer.py --batch_size 40 --model_dir ernie_float32 --backend pp-trt --use_fp16 True
python infer.py --batch_size 40 --model_dir save_ernie3_afqmc_new_cablib --backend pp-trt

# TRT
python infer.py --batch_size 40 --model_dir ernie_float32 --backend trt
python infer.py --batch_size 40 --model_dir ernie_float32 --backend trt --use_fp16 True
python infer.py --batch_size 40 --model_dir save_ernie3_afqmc_new_cablib --backend trt --use_fp16 True

# CPU PP
python infer.py --batch_size 40 --cpu_num_threads 10 --model_dir ernie_float32 --backend pp --device cpu
python infer.py --batch_size 40 --cpu_num_threads 10 --model_dir save_ernie3_afqmc_new_cablib --backend pp --device cpu

# CPU ORT
python infer.py --batch_size 40 --cpu_num_threads 10 --model_dir ernie_float32/ --backend ort --device cpu
python infer.py --batch_size 40 --cpu_num_threads 10 --model_dir save_ernie3_afqmc_new_cablib/ --backend ort --device cpu
