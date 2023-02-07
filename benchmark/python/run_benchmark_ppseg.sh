echo "[FastDeploy]    Running PPseg benchmark..."

num_of_models=$(ls -d ppseg_model/* | wc -l)

counter=1
for model in $(ls -d ppseg_model/* )
do
    echo "[Benchmark-PPseg] ${counter}/${num_of_models} $model ..."
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 1 --iter_num 2000 --backend ort --enable_collect_memory_info True
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend ort --enable_collect_memory_info True
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 1 --iter_num 2000 --backend paddle --enable_collect_memory_info True
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend paddle --enable_collect_memory_info True
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 1 --iter_num 2000 --backend ov --enable_collect_memory_info True
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend ov --enable_collect_memory_info True
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend ort --enable_collect_memory_info True
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend paddle --enable_collect_memory_info True
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend paddle_trt --enable_collect_memory_info True
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend paddle_trt --enable_trt_fp16 True --enable_collect_memory_info True
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend trt --enable_collect_memory_info True
    python benchmark_ppseg.py --model $model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend trt --enable_trt_fp16 True --enable_collect_memory_info True
    counter=$(($counter+1))
    step=$(( $counter % 1 ))
    if [ $step = 0 ]
    then
        wait
    fi
done

wait

rm -rf result_ppseg.txt
touch result_ppseg.txt
cat ppseg_model/*.txt >> ./result_ppseg.txt

python convert_info.py --txt_path result_ppseg.txt --domain ppseg --enable_collect_memory_info True
