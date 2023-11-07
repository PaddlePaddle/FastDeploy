#!/bin/bash

pip install wget
python3 -u CI.py
result=$?
if [ ${result} -eq 0 ];then
  echo "通过测试"
else
  echo "测试失败"
fi
echo "具体结果如下："
cat results.txt
exit $result

