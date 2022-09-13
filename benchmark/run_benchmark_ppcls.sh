echo "[FastDeploy]    Running PPcls benchmark..."
find . -name "*.txt" | xargs rm -rf

num_of_models=$(ls -d ppcls_model/* | wc -l)

counter=1
for model in $(ls -d ppcls_model/* )
do
    echo "[Benchmark-PPcls] ${counter}/${num_of_models} $model ..."
    python benchmark_ppcls.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 1 --iter_num 2000 --backend ort
    python benchmark_ppcls.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend ort
    python benchmark_ppcls.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 1 --iter_num 2000 --backend paddle
    python benchmark_ppcls.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend paddle
    python benchmark_ppcls.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 1 --iter_num 2000 --backend ov
    python benchmark_ppcls.py --model $model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend ov
    python benchmark_ppcls.py --model $model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend ort
    python benchmark_ppcls.py --model $model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend paddle
    python benchmark_ppcls.py --model $model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend trt
    python benchmark_ppcls.py --model $model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend trt --enable_trt_fp16 True
    counter=$(($counter+1))
    step=$(( $counter % 1 ))
    if [ $step = 0 ]
    then
        wait
    fi
done

wait

rm -rf result_ppcls.txt
touch result_ppcls.txt
cat ppcls_model/*.txt >> ./result_ppcls.txt

python convert_info.py --txt_path result_ppcls.txt --domain ppcls
