echo "[FastDeploy]    Running PPcls benchmark..."

num_of_models=$(ls -d ppcls_model/ | wc -l)

counter=1
for model in $(ls -d ppcls_model/ )
do
    echo "[Benchmark-PPcls] ${counter}/${num_of_models} $model ..."
    python benchmark_cls.py --model ppcls_model/$model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 1 --iter_num 2000 --backend ort
    python benchmark_cls.py --model ppcls_model/$model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend ort
    python benchmark_cls.py --model ppcls_model/$model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 1 --iter_num 2000 --backend paddle
    python benchmark_cls.py --model ppcls_model/$model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend paddle
    python benchmark_cls.py --model ppcls_model/$model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 1 --iter_num 2000 --backend ov
    python benchmark_cls.py --model ppcls_model/$model --image ILSVRC2012_val_00000010.jpeg --cpu_num_thread 8 --iter_num 2000 --backend ov
    python benchmark_cls.py --model ppcls_model/$model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend ort
    python benchmark_cls.py --model ppcls_model/$model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend paddle
    python benchmark_cls.py --model ppcls_model/$model --image ILSVRC2012_val_00000010.jpeg --device gpu --iter_num 2000 --backend trt
    counter=$(($counter+1))
    step=$(( $counter % 1 ))
    if [ $step = 0 ]
    then
        wait
    fi
done

wait

rm -rf result.txt
touch result.txt
cat ppcls_model/*.txt >> ./result.txt

# number_lines=$(cat result.txt | wc -l)
# failed_line=$(grep -o "Failed"  result.txt|wc -l)
# zero=0
# if [ $failed_line -ne $zero ]
# then
#     echo "[ERROR] There are $number_lines results in result.txt, but failed number of models is $failed_line."
#     exit -1
# else
#     echo "All Succeed!"
# fi
