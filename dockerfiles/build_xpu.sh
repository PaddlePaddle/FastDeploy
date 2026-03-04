cp ../requirements.txt ./
PRODUCT_NAME=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-xpu:ci
docker build --no-cache -t ${PRODUCT_NAME} -f Dockerfile.xpu . \
    --network host \
    --build-arg HTTP_PROXY=${http_proxy} \
    --build-arg HTTPS_PROXY=${http_proxy} \
