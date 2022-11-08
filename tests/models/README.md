# 添加模型单测


所有模型统一使用`runtime_config.py`中的RuntimeOption进行配置

```
import runtime_config as rc


model = fd.vision.XXX(..., runtime_option=rc.test_option)
```


验证For循环跑2+次与Baseline结果符合预期 
