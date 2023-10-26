if [ $# -ne 2 ]; then
    echo "[ERROR] Usage: bash gen_serving_models.sh source_model_dir target_model_dir"
    exit 1
fi

source_model=$1
target_model=$2

if [ -d "$target_model" ]; then
    echo "[ERROR] The target model dir: $target_model is already exists, please remove it first."
    exit 1
fi

if [ -d "$source_model" ]; then
    echo "[INFO] Will generate a serving model based on $source_model."
else
    echo "[ERROR] The source model dir: $source_model is not exist."
    exit 1
fi

echo "[INFO] Creating target model dir: $target_model ..."
mkdir -p $target_model

rm -rf triton-server-template.tar
wget https://bj.bcebos.com/fastdeploy/llm/triton-server-template.tar

tar xvf triton-server-template.tar

echo "[INFO] Copy source file to target directory..."
cp -r triton-server-template/* $target_model
cp $source_model/* $target_model/model-aistudio/1/
echo "[INFO] Finished, the serving model is saved in $target_model."
