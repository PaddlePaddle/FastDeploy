# wget https://bj.bcebos.com/fastdeploy/benchmark/uie/reimbursement_form_data.txt
# wget https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz

# GPU
## FP32 Model
python benchmark_uie.py --model_dir uie-base --data_path reimbursement_form_data.txt --backend pp --device gpu
python benchmark_uie.py --model_dir uie-base --data_path reimbursement_form_data.txt --backend ort --device gpu
python benchmark_uie.py --model_dir uie-base --data_path reimbursement_form_data.txt --backend pp-trt --device gpu
python benchmark_uie.py --model_dir uie-base --data_path reimbursement_form_data.txt --backend trt --device gpu

## INT8 Model
python benchmark_uie.py --model_dir uie_bs1_lr1e-5_qat_final_format_4inputs --data_path reimbursement_form_data.txt --backend pp-trt --device gpu
python benchmark_uie.py --model_dir uie_bs1_lr1e-5_qat_final_format_4inputs --data_path reimbursement_form_data.txt --backend trt --device gpu

# CPU
## FP32 Model
python benchmark_uie.py --model_dir uie-base --data_path reimbursement_form_data.txt --backend pp --device cpu
python benchmark_uie.py --model_dir uie-base --data_path reimbursement_form_data.txt --backend ort --device cpu
python benchmark_uie.py --model_dir uie-base --data_path reimbursement_form_data.txt --backend openvino --device cpu

## INT8 Model

python benchmark_uie.py --model_dir uie_bs1_lr1e-5_qat_final_format_4inputs --data_path reimbursement_form_data.txt --backend pp --device cpu
python benchmark_uie.py --model_dir uie_bs1_lr1e-5_qat_final_format_4inputs --data_path reimbursement_form_data.txt --backend ort --device cpu
