PRODUCT_NAME='ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:fastdeploy-ciuse-cuda126'
#proxy='agent.baidu.com:8188'
cp ../../requirements.txt ./

docker build -t ${PRODUCT_NAME} -f Dockerfile.ci . \
    --network host
 #   --build-arg HTTP_PROXY=${proxy} \
 #   --build-arg HTTPS_PROXY=${proxy} \
   # --build-arg ftp_proxy=${proxy}
