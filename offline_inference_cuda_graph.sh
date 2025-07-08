for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAG_SAMPLING_CLASS=rejection 
export FD_DEBUG=1
export PYTHONPATH=/root/paddlejob/workspace/env_run/output/gongshaotian/FastDeploy:$PYTHONPATH
rm -rf log

python offline_inference_cuda_graph.py