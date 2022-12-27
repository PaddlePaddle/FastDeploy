# wget https://bj.bcebos.com/fastdeploy/benchmark/uie/reimbursement_form_data.txt
# wget https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz
# tar xvfz uie-base.tgz
# GPU
echo "-------------------------------GPU Benchmark---------------------------------------"
python benchmark_uie.py --log_interval 100 --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend pp --device gpu
python benchmark_uie.py --log_interval 100 --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend ort --device gpu
python benchmark_uie.py --log_interval 100 --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend pp-trt --device gpu --use_fp16 False
python benchmark_uie.py --log_interval 100 --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend trt --device gpu --use_fp16 False
python benchmark_uie.py --log_interval 100 --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend pp-trt --device gpu --use_fp16 True
python benchmark_uie.py --log_interval 100 --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend trt --device gpu --use_fp16 True
echo "-----------------------------------------------------------------------------------"

# CPU
echo "-------------------------------CPU Benchmark---------------------------------------"
for cpu_num_threads in 1 2 4 8 16;
do
  python benchmark_uie.py --log_interval 100 --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend pp --device cpu --cpu_num_threads ${cpu_num_threads}
  python benchmark_uie.py --log_interval 100 --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend ort --device cpu --cpu_num_threads ${cpu_num_threads}
  python benchmark_uie.py --log_interval 100 --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend openvino --device cpu --cpu_num_threads ${cpu_num_threads}
done
echo "-----------------------------------------------------------------------------------"
