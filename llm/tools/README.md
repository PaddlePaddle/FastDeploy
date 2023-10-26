# 工具脚本

- text_max_batch_size.py 用于测试服务可承载的最大batch size，使用方式`python test_max_batch_size.py model_dir batch_size max_seq_len max_dec_len`，例如`python test_max_batch_size.py chatglm-6b-fp16 4 2048 1024`。测试方式为不断加大batch size，看是否能正常预测。直到出现显存不足错误。

- gen_serving_model.sh 由于服务化部署时，需要按照Triton目录格式准备部署模型，此脚本用于从PaddleNLP导出模型转为服务部署模型，使用方式`bash gen_serving_model.sh paddlenlp_exported_model serving_model_path`

- benchmark.py 用于测试服务性能
