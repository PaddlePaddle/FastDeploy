

api_server_pids=$(ps auxww | grep "api_server" | grep -v grep | awk '{print $2}')
echo 'end api server pids:'
echo $api_server_pids

for pid in $api_server_pids; do
    child_pids=$(ps -ef | grep $pid | grep -v grep | awk '{print $2}')
    echo $child_pids
    for in_pid in ${child_pids[@]}; do
        kill -9 ${in_pid}
    done
    echo 'end uvicorn multi workers'
done


fastdeploy_inferernce_pids=$(ps auxww | grep "fastdeploy" | grep -v grep | awk '{print $2}')
echo $fastdeploy_inferernce_pids
for in_pid in ${fastdeploy_inferernce_pids[@]}; do
    kill -9 ${in_pid}
done
echo 'end fastDeploy inference pids'
