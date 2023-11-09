import sys
import json
import os
f1 = open(sys.argv[2], "w", newline='')
for i in range(len(os.listdir(sys.argv[1]))):
    result = ""
    with open("{}/{}".format(sys.argv[1], i), 'r') as f:
        res = dict()
        for line in f:
            if line.startswith("task_id="):
                continue
            data = eval(line.strip())
            result += data[1]
        res[i] = result
        f1.write("{}".format(res))
        f1.write("\n")
