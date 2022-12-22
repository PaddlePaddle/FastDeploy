import requests
import json

url = 'http://127.0.0.1:8000/fd/ppyoloe'
headers = {"Content-Type": "application/json"}

data = {"data": {"image": "qwertyuiopasdfghjkl"}, "parameters": {}}

r = requests.post(url=url, headers=headers, data=json.dumps(data))
print(r.text)
# result_json = json.loads(r.text)
# print(result_json)
