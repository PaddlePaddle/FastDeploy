# wget https://bj.bcebos.com/fastdeploy/benchmark/uie/reimbursement_form_data.txt
# wget https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz
# tar xvfz uie-base.tgz

DEVICE_ID=0

echo "[FastDeploy]    Running UIE benchmark..."

# GPU
echo "-------------------------------GPU Benchmark---------------------------------------"
python benchmark_uie.py --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend paddle --device_id $DEVICE_ID --device gpu --enable_collect_memory_info True
python benchmark_uie.py --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend ort --device_id $DEVICE_ID --device gpu --enable_collect_memory_info True
python benchmark_uie.py --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend paddle_trt --device_id $DEVICE_ID --device gpu --enable_trt_fp16 False --enable_collect_memory_info True
python benchmark_uie.py --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend trt --device_id $DEVICE_ID --device gpu --enable_trt_fp16 False --enable_collect_memory_info True
python benchmark_uie.py --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend paddle_trt --device_id $DEVICE_ID --device gpu --enable_trt_fp16 True --enable_collect_memory_info True
python benchmark_uie.py --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend trt --device_id $DEVICE_ID --device gpu --enable_trt_fp16 True --enable_collect_memory_info True
echo "-----------------------------------------------------------------------------------"

# CPU
echo "-------------------------------CPU Benchmark---------------------------------------"
for cpu_num_threads in 1 8;
do
  python benchmark_uie.py --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend paddle --device cpu --cpu_num_threads ${cpu_num_threads} --enable_collect_memory_info True
  python benchmark_uie.py --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend ort --device cpu --cpu_num_threads ${cpu_num_threads} --enable_collect_memory_info True
  python benchmark_uie.py --epoch 5 --model_dir uie-base --data_path reimbursement_form_data.txt --backend ov --device cpu --cpu_num_threads ${cpu_num_threads} --enable_collect_memory_info True
done
echo "-----------------------------------------------------------------------------------"
