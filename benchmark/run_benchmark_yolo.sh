echo "[FastDeploy]    Running Yolo benchmark..."

num_of_models=$(ls -d yolo_model/* | wc -l)

counter=1
for model in $(ls -d yolo_model/* )
do
    echo "[Benchmark-Yolo] ${counter}/${num_of_models} $model ..."
    python benchmark_yolo.py --model $model --image 000000014439.jpg --cpu_num_thread 8 --iter_num 1000 --backend paddle --enable_collect_memory_info True
    python benchmark_yolo.py --model $model --image 000000014439.jpg --cpu_num_thread 8 --iter_num 1000 --backend ort --enable_collect_memory_info True
    python benchmark_yolo.py --model $model --image 000000014439.jpg --cpu_num_thread 8 --iter_num 1000 --backend ov --enable_collect_memory_info True
    python benchmark_yolo.py --model $model --image 000000014439.jpg --device gpu --iter_num 1000 --backend ort --enable_collect_memory_info True
    python benchmark_yolo.py --model $model --image 000000014439.jpg --device gpu --iter_num 1000 --backend paddle --enable_collect_memory_info True
    python benchmark_yolo.py --model $model --image 000000014439.jpg --device gpu --iter_num 1000 --backend paddle_trt --enable_collect_memory_info True
    python benchmark_yolo.py --model $model --image 000000014439.jpg --device gpu --iter_num 1000 --backend paddle_trt --enable_trt_fp16 True --enable_collect_memory_info True
    python benchmark_yolo.py --model $model --image 000000014439.jpg --device gpu --iter_num 1000 --backend trt --enable_collect_memory_info True
    python benchmark_yolo.py --model $model --image 000000014439.jpg --device gpu --iter_num 1000 --backend trt --enable_trt_fp16 True --enable_collect_memory_info True
    counter=$(($counter+1))
    step=$(( $counter % 1 ))
    if [ $step = 0 ]
    then
        wait
    fi
done

wait

rm -rf result_yolo.txt
touch result_yolo.txt
cat yolo_model/*.txt >> ./result_yolo.txt

python convert_info.py --txt_path result_yolo.txt --domain yolo --enable_collect_memory_info True
