export devices=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch --gpus ${devices} convert_ep_state_from_tp8.py --model_dir /path/to/model_dir