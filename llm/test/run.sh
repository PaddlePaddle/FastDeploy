#!/bin/bash

current_directory=$PWD

#环境安装 主要是wget的安装和paddlenlp算子
pip install wget
cd ${paddlenlp}/csrc
${py_version} setup_cuda.py install --user



#模型文件下载
cd $current_directory
#下载解压预存结果
NLP_name='paddlenlp_llm_results'
FD_name='fastdeploy_llm_dynamic_batching_results'
wget https://bj.bcebos.com/paddle2onnx/third_libs/${NLP_name}.tar
wget https://bj.bcebos.com/paddle2onnx/third_libs/${FD_name}.tar
tar -xvf https://bj.bcebos.com/paddle2onnx/third_libs/${NLP_name}.tar
tar -xvf https://bj.bcebos.com/paddle2onnx/third_libs/${FD_name}.tar
mkdir pre_result
mv ${NLP_name}/* pre_result/
mv ${FD_name}/* pre_result/
rm -f ${NLP_name}.tar
rm -f ${FD_name}.tar
#下载测试文件
wget -O inputs_base.jsonl https://bj.bcebos.com/paddle2onnx/third_libs/inputs_63.jsonl
wget -O inputs_precache.jsonl https://bj.bcebos.com/paddle2onnx/third_libs/ptuning_inputs.json
#下载precache文件以及导出静态模型
export_model_name=('linly-ai/chinese-llama-2-7b' 'THUDM/chatglm-6b' 'bellegroup/belle-7b-2m')
precache_url=('https://bj.bcebos.com/fastdeploy/llm/llama-7b-precache.npy' 'https://bj.bcebos.com/fastdeploy/llm/chatglm-6b-precache.npy' 'https://bj.bcebos.com/fastdeploy/llm/bloom-7b-precache.npy')
noptuning_model_name=('llama-7b-fp16' 'chatglm-6b-fp16' 'belle-7b-2m-fp16')
ptuning_model_name=('llama-7b-ptuning-fp16' 'chatglm-6b-ptuning-fp16' 'belle-7b-2m-ptuning-fp16')
target_name='task_prompt_embeddings.npy'
for((i=0;i<${#precache_url[*]};i++));do
  mkdir -p precache_${ptuning_model_name[i]}/8-test/1
  cd precache_${ptuning_model_name[i]}/8-test/1
  wget -O ${target_name} ${precache_url[i]}
  cd $current_directory
done
mkdir inference_model
cd ${paddlenlp}/llm
for((i=0;i<${#export_model_name[*]};i++));do
${py_version} export_model.py --model_name_or_path ${export_model_name[i]} --output_path  ${current_directory}/inference_model/${noptuning_model_name[i]} --dtype float16 --inference_model
${py_version} export_model.py --model_name_or_path ${export_model_name[i]} --output_path  ${current_directory}/inference_model/${ptuning_model_name[i]} --dtype float16 --inference_model --export_precache 1
done
cd $current_directory
#开启测试
${py_version} -u ci.py
result=$?
if [ $result -eq 0 ];then
  echo "通过测试"
else
  echo "测试失败"
fi
echo "具体结果如下："
cat results.txt
exit $result
