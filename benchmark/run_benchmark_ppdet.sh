echo "[FastDeploy]    Running PPdet benchmark..."
find . -name "*.txt" | xargs rm -rf

num_of_models=$(ls -d ppdet_model/* | wc -l)

counter=1
for model in $(ls -d ppdet_model/* )
do
    echo "[Benchmark-PPdet] ${counter}/${num_of_models} $model ..."
    python benchmark_ppdet.py --model $model --image 000000014439.jpg --cpu_num_thread 1 --iter_num 2000 --backend ort
    python benchmark_ppdet.py --model $model --image 000000014439.jpg --cpu_num_thread 8 --iter_num 2000 --backend ort
    python benchmark_ppdet.py --model $model --image 000000014439.jpg --cpu_num_thread 1 --iter_num 2000 --backend paddle
    python benchmark_ppdet.py --model $model --image 000000014439.jpg --cpu_num_thread 8 --iter_num 2000 --backend paddle
    python benchmark_ppdet.py --model $model --image 000000014439.jpg --cpu_num_thread 1 --iter_num 2000 --backend ov
    python benchmark_ppdet.py --model $model --image 000000014439.jpg --cpu_num_thread 8 --iter_num 2000 --backend ov
    python benchmark_ppdet.py --model $model --image 000000014439.jpg --device gpu --iter_num 2000 --backend ort
    python benchmark_ppdet.py --model $model --image 000000014439.jpg --device gpu --iter_num 2000 --backend paddle
    python benchmark_ppdet.py --model $model --image 000000014439.jpg --device gpu --iter_num 2000 --backend trt
    python benchmark_ppdet.py --model $model --image 000000014439.jpg --device gpu --iter_num 2000 --backend trt --enable_trt_fp16 True
    counter=$(($counter+1))
    step=$(( $counter % 1 ))
    if [ $step = 0 ]
    then
        wait
    fi
done

wait

rm -rf result_ppdet.txt
touch result_ppdet.txt
cat ppdet_model/*.txt >> ./result_ppdet.txt

python convert_info.py --txt_path result_ppdet.txt --domain ppdet
