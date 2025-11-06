pkill -9 -f python
pkill -9 -f fastdeploy
pkill -f -9 gunicorn

if redis-cli ping >/dev/null 2>&1; then
    redis-cli shutdown
fi
