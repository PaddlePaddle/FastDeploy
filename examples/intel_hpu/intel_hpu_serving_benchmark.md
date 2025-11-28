# Intel HPU serving benchmark

## 1. start server
```bash
./benchmark_paddle_hpu_server.sh
```
In the script, you can use HPU_VISIBLE_DEVICES to select hpu card.

## 2. run benchmark cli
```bash
./benchmark_paddle_hpu_cli.sh
```

## 3. parse logs
```python
python parse_benchmark_logs.py benchmark_fastdeploy_logs/[the targeted folder]
```
The performance data will be saved as a CSV file.

## 4. analyse logs
```python
python draw_benchmark_data.py benchmark_fastdeploy_logs/[the targeted folder]
```
The script will save the model execution times and batch tokens as a CSV file and plot them in a graph.
