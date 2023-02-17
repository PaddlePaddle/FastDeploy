TARGET_DEVICE=ascend

model_dir=`ls ./models/`

for MODEL_NAME in $model_dir
do
    python eval.py --model ./models/$MODEL_NAME  --image None --device $TARGET_DEVICE 2>&1 | tee ./log/${MODEL_NAME}_acc.log
done
