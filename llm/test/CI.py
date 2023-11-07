#命令行提供：1.PaddleNLP/llm的路径   2.Fastdeploy/llm的路径   3.关于存放（Paddlenlp结果和FD_DY结果的数据文件）
#存放的数据文件包括：NLP-llama-7b-fp16-bs1,NLP-llama-7b-fp16-bs4,NLP-llama-7b-ptuning-fp16-bs1,NLP-llama-7b-ptuning-fp16-bs4,NLP-llama-7b-ptuning-fp16-bs1-noprecache,NLP-llama-7b-ptuning-fp16-bs4-noprecache,
#NLP-belle-7b-2m-fp16-bs1,NLP-belle-7b-2m-fp16-bs4,NLP-belle-7b-2m-ptuning-fp16-bs1,NLP-belle-7b-2m-ptuning-fp16-bs4,NLP-belle-7b-2m-ptuning-fp16-bs1-noprecache,NLP-belle-7b-2m-ptuning-fp16-bs4-noprecache
#FD-llama-7b-fp16-bs4-dy,FD-llama-7b-ptuning-fp16-bs4-dy,FD-llama-7b-ptuning-fp16-bs4-dy-noprecache,FD-chatglm-6b-fp16-bs4-dy,FD-chatglm-6b-ptuning-fp16-bs4-dy,FD-chatglm-6b-ptuning-fp16-bs4-dy-noprecache,FD-belle-7b-2m-fp16-bs4-dy,FD-belle-7b-2m-ptuning-fp16-bs4-dy,FD-belle-7b-2m-ptuning-fp16-bs4-dy-noprecache

#test_serving,read_serving以及compute_diff.py与IC打包到同一个文件夹
import os
import sys
import wget
import compute_diff
def main():
    #安装PaddleNLP/FastDeploy包
    current_file_path = os.path.abspath(os.getcwd())
    os.system(command="git clone https://github.com/PaddlePaddle/PaddleNLP.git")
    os.system(command="git clone -b llm https://github.com/PaddlePaddle/FastDeploy.git")
    os.chdir(f'{current_file_path}/PaddleNLP')
    os.system('python3 setup.py bdist_wheel')
    wheel_name = os.listdir(f'{current_file_path}/PaddleNLP/dist/')[0]
    os.chdir(f"{current_file_path}/PaddleNLP/dist")
    os.system(f'pip install {wheel_name}')
    os.chdir(f'{current_file_path}/PaddleNLP/csrc')
    os.system(f'python3 setup_cuda.py install --user')
    os.system(f"{current_file_path}")


    #以下跑程序都用绝对路径
    inference_model_path=f'{current_file_path}/inference_model' #推理模型导出存放文件
    pre_result_path=f'{current_file_path}/pre_result'  #预存对比结果的文件
    #输出表格数据的文件路径
    out_path=f'{current_file_path}/results.txt'
    if os.path.exists(out_path):    #原本存在，则删除，后面写文件会创建一个新的文件夹
        os.remove(out_path)
    #从网上下载测试结果，分别为NLP预存tar包和FD预存tar包，存入pre_result_path
    if os.path.exists(pre_result_path):
        os.system(command=f"rm -rf {pre_result_path}")
    os.mkdir(pre_result_path)
    NLP_name='paddlenlp_llm_results'
    FD_name='fastdeploy_llm_dynamic_batching_results'
    NLP_url=f'https://bj.bcebos.com/paddle2onnx/third_libs/{NLP_name}.tar'
    FD_url=f'https://bj.bcebos.com/paddle2onnx/third_libs/{FD_name}.tar'

    wget.download(NLP_url)
    wget.download(FD_url)
    os.system(command=f"tar -xvf {NLP_name}.tar ")
    os.system(command=f"tar -xvf {FD_name}.tar ")
    os.system(command=f"mv {NLP_name}/* {pre_result_path}")
    os.system(command=f"mv {FD_name}/* {pre_result_path}")
    os.system(command=f"rm -f {NLP_name}.tar")
    os.system(command=f"rm -f {FD_name}.tar")
    #准备工作，导出模型
    export_model_name= ['linly-ai/chinese-llama-2-7b','THUDM/chatglm-6b','bellegroup/belle-7b-2m']
    noptuning_model_name=['llama-7b-fp16','chatglm-6b-fp16','belle-7b-2m-fp16']
    ptuning_model_name = ['llama-7b-ptuning-fp16','chatglm-6b-ptuning-fp16','belle-7b-2m-ptuning-fp16']
    num_model=len(export_model_name)
    #存放模型的绝对路径
    noptuning_model_path_list = []
    ptuning_model_path_list = []
    #非P-Tuning导出以及P-Tuning导出
    #判断存放模型文件是否存在
    if os.path.exists(inference_model_path):
        os.system(command=f"rm -rf {inference_model_path}")
    os.mkdir(inference_model_path)
    for i in range(num_model):
        noptuning_model_path=os.path.join(inference_model_path,f"{noptuning_model_name[i]}")
        ptuning_model_path=os.path.join(inference_model_path,f"{ptuning_model_name[i]}")
        noptuning_model_path_list.append(noptuning_model_path)
        ptuning_model_path_list.append(ptuning_model_path)
        os.chdir(f"{current_file_path}/PaddleNLP/llm")
        #非P-Tuning
        if not os.path.exists(noptuning_model_path):
            os.system(command=f"python3 export_model.py --model_name_or_path {export_model_name[i]} --output_path  {noptuning_model_path} --dtype float16 --inference_model")
        #P-Tuning
        if not os.path.exists(ptuning_model_path):
            os.system(command=f"python3 export_model.py --model_name_or_path {export_model_name[i]} --output_path  {ptuning_model_path} --dtype float16 --inference_model --export_precache 1")
        #模型会导出到 PaddleNLP/llm/inference_model/

    #下载precache
    #在Fastdeploy/llm中创建三个文件夹，存放三个模型的precache

    precache_url=['https://bj.bcebos.com/fastdeploy/llm/llama-7b-precache.npy','https://bj.bcebos.com/fastdeploy/llm/chatglm-6b-precache.npy','https://bj.bcebos.com/fastdeploy/llm/bloom-7b-precache.npy']
    target_name='task_prompt_embeddings.npy'
    precache_path_list=[]
    for i in range(num_model):
        precache_path=f"{current_file_path}/precache_{ptuning_model_name[i]}"
        precache_path_list.append(precache_path)
        precache_path_FD=os.path.join(precache_path,'8-test','1')
        if os.path.exists(precache_path_FD):
            continue
        else:
            os.system(command=f"mkdir -p {precache_path_FD}")
            wget.download(precache_url[i],out=os.path.join(precache_path,'8-test','1',target_name))



    #下载测试文件
    inputs_url = 'https://bj.bcebos.com/paddle2onnx/third_libs/inputs_63.jsonl'
    inputs_name=f'{current_file_path}/inputs_base.jsonl'
    inputs_path=inputs_name
    if os.path.exists(inputs_path):
        os.system(command=f"rm -f {inputs_path}")
    wget.download(inputs_url,out=inputs_path)
    inputs_PT_url= 'https://bj.bcebos.com/paddle2onnx/third_libs/ptuning_inputs.json'
    inputs_PT_name = f'{current_file_path}/inputs_precache.jsonl'
    inputs_PT_path=inputs_PT_name
    if os.path.exists(inputs_PT_path):
        os.system(command=f"rm -f {inputs_PT_path}")
    wget.download(inputs_PT_url,out=inputs_PT_path)

    #进入Fastdeploy/llm进行测试
    #分三个list进行结果存储(只存储一个模型的一行）
    no_PT=[]#非P_Tuning
    PT=[]     #P_Tuning
    pre_PT=[]   #P-Tuning with precache

    #分三种情况   bs=1  bs=4   bs=4stop=2
    opts=['bs1','bs4','bs4-dy']

    #清空共享内存
    os.system(command='rm -rf /dev/shm')
    #创建res文件进行结果存储,若已存在文件则将文件结果删除
    res_path=f'{current_file_path}/FastDeploy/llm/res'
    if os.path.exists(res_path):
        os.system(command=f"rm -f {res_path}/*")
    else:
        os.mkdir(res_path)
    #删除运行时模型输出文件
    os.system(command=f"rm -f real_time_save.temp_ids_rank_0_step*")
    #创建存放FD测试结果文件夹
    FD_result_path=f'{current_file_path}/FD_result'
    if os.path.exists(FD_result_path):
        os.system(command=f"rm -rf {FD_result_path}")
    os.mkdir(FD_result_path)
    #测试非ptuning并保存diff率
    # python3 test_serving.py /work/model_pkg/belle-7b-2m-fp16 inputs_63.jsonl 4 1
    # python3 read_serving.py res fd_result/llama-6b-fp16-bs1.txt
    # python3 print_diff.py  nlp_result/chatglm-6b-fp16-bs4.txt  fd_result/chatglm-6b-fp16-bs4-dy.txt
    # NLP-belle-7b-2m-fp16-bs1

    batch_size=[1,4,4]
    disdy=[1,1,0]
    mopt=['NLP','NLP','FD']
    bug_flag=0
    #总共需要三个维度，模型名称，模型类型（非ptuning,ptuning without precache,ptuning with precache),参数设置（bs=1,bs=4,bs=4动插）
    os.system(f'cp {current_file_path}/test_serving.py {current_file_path}/FastDeploy/llm/test_serving.py')
    os.system(f'cp {current_file_path}/read_serving.py {current_file_path}/FastDeploy/llm/read_serving.py')

    #写入文件表头,获取非P-Tuning情况
    with open(out_path,'a+') as f:
        f.write("非PTuning FP16 model test\n")
        #f.write("模型\t\tbs=1（与PaddleNLP对比)\t\tbs=4（与PaddleNLP对比)\t\tbs=4 stop=2(动态插入，与FD上一版本进行对比\n")
        #f.write('%-24s%-24s%-24s%-24s' % ("模型", "bs=1（与PaddleNLP对比)", "bs=4（与PaddleNLP对比)", "bs=4 stop=2(动态插入，与FD上一版本进行对比"))
        f.write('%-30s%-30s%-30s%-30s\n' % ("model", "bs=1(compare with PaddleNLP)", "bs=4(compare with PaddleNLP)","bs=4 stop=2(compare with FD)"))
    os.chdir(f"{current_file_path}/FastDeploy/llm")
    for model_index in range(len(noptuning_model_path_list)):  #遍历模型路径
        for i in range(3): #遍历参数设置
            os.system(f"python3 test_serving.py {noptuning_model_path_list[model_index]} {inputs_path} {batch_size[i]} {disdy[i]} 0 0 {res_path}")  #倒数二三个参数表示ptuning/precache
            os.system(f"python3 read_serving.py {res_path} {FD_result_path}/{noptuning_model_name[model_index]}-{opts[i]}.txt")
            file1=os.path.join(pre_result_path,f"{mopt[i]}-{noptuning_model_name[model_index]}-{opts[i]}.txt")
            file2=f"{FD_result_path}/{noptuning_model_name[model_index]}-{opts[i]}.txt"
            is_diff,diff_rate = compute_diff.get_diff(file1, file2)
            if is_diff:
                bug_flag=1
            no_PT.append(diff_rate)
            os.system(command=f"rm -f {res_path}/*")
            os.system(command=f"rm -f real_time_save.temp_ids_rank_0_step_*")
            os.system(command="rm -rf /dev/shm/*")
        with open(out_path, 'a+') as f:
            #f.write(f"{noptuning_model_name[model_index]}\t\t{no_PT[0]}\t\t{no_PT[1]}\t\t{no_PT[2]}\n")
            f.write('%-30s%-30s%-30s%-30s\n' % (noptuning_model_name[model_index], no_PT[0], no_PT[1], no_PT[2]))

        no_PT = []

    with open(out_path, 'a+') as f:
        f.write("\n")


    #写入文件表头
    with open(out_path,'a+') as f:
        f.write("PTuning FP16 model test\n")
        #f.write("模型\t\t是否传precache\t\tbs=1（与PaddleNLP对比)\t\tbs=4（与PaddleNLP对比)\t\tbs=4 stop=2(动态插入，与FD上一版本进行对比\n")
        f.write('%-30s%-30s%-30s%-30s%-30s\n' % ("model","whether send precache","bs=1(compare with PaddleNLP)","bs=4(compare with PaddleNLP)","bs=4 stop=2(compare with FD)"))

    #获取P-Tuning without precache
    for model_index in range(len(ptuning_model_path_list)):  #遍历模型名称
        for i in range(3): #遍历参数设置
            os.system(f"python3 test_serving.py {ptuning_model_path_list[model_index]} {inputs_path} {batch_size[i]} {disdy[i]} 1 0 {res_path}")  #倒数二三个参数表示ptuning/precache
            os.system(f"python3 read_serving.py {res_path} {FD_result_path}/{ptuning_model_name[model_index]}-{opts[i]}-noprecache.txt")
            file1=os.path.join(pre_result_path,f"{mopt[i]}-{ptuning_model_name[model_index]}-{opts[i]}-noprecache.txt")
            file2=f"{FD_result_path}/{ptuning_model_name[model_index]}-{opts[i]}-noprecache.txt"
            is_diff, diff_rate = compute_diff.get_diff(file1, file2)
            if is_diff:
                bug_flag = 1
            PT.append(diff_rate)
            os.system(command=f"rm -f {res_path}/*")
            os.system(command=f"rm -f real_time_save.temp_ids_rank_0_step_*")
            os.system(command="rm -rf /dev/shm/*")
        with open(out_path, 'a+') as f:
            #f.write(f"{ptuning_model_name[model_index]}\t\t否\t\t{PT[0]}\t\t{PT[1]}\t\t{PT[2]}\n")
            f.write('%-30s%-30s%-30s%-30s%-30s\n' % (ptuning_model_name[model_index], 'no', PT[0], PT[1], PT[2]))
        PT = []


    #获取P-Tuning with precache

    for model_index in range(len(ptuning_model_path_list)):  #遍历模型名称
        for i in range(3): #遍历参数设置
            os.system(f"python3 test_serving.py {ptuning_model_path_list[model_index]} {inputs_PT_path} {batch_size[i]} {disdy[i]} 1 1 {res_path} {precache_path_list[model_index]}")  #倒数二三个参数表示ptuning/precache
            os.system(f"python3 read_serving.py {res_path} {FD_result_path}/{ptuning_model_name[model_index]}-{opts[i]}.txt")
            file1=os.path.join(pre_result_path,f"{mopt[i]}-{ptuning_model_name[model_index]}-{opts[i]}.txt")
            file2=f"{FD_result_path}/{ptuning_model_name[model_index]}-{opts[i]}.txt"
            is_diff, diff_rate = compute_diff.get_diff(file1, file2)
            if is_diff:
                bug_flag = 1
            pre_PT.append(diff_rate)
            os.system(command=f"rm -f {res_path}/*")
            os.system(command=f"rm -f real_time_save.temp_ids_rank_0_step_*")
            os.system(command="rm -rf /dev/shm/*")

        with open(out_path, 'a+') as f:
            #f.write(f"{ptuning_model_name[model_index]}\t\t是\t\t{pre_PT[0]}\t\t{pre_PT[1]}\t\t{pre_PT[2]}\n")
            f.write('%-30s%-30s%-30s%-30s%-30s\n' % (ptuning_model_name[model_index], 'yes', pre_PT[0], pre_PT[1], pre_PT[2]))

        pre_PT = []
    os.system(f"{current_file_path}")

    return bug_flag


if __name__=="__main__":
    main()























