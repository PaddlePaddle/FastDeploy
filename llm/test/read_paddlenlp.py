import json
import sys
 
f1 = open(sys.argv[2], "w")
with open(sys.argv[1], "r") as f:
    for i, line in enumerate(f):
        data = eval(line.strip())
        res = {i: data["output"]}
        f1.write("{}".format(res))
        f1.write("\n")
