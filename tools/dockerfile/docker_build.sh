PRODUCT_NAME='ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:fastdeploy-ciuse-cuda126'
cp ../../requirements.txt ./
cp ../../scripts/unittest_requirement.txt ./
docker build -t ${PRODUCT_NAME} -f Dockerfile.ci . \
    --network host
