import openai
import subprocess
import os
import time
import signal

model = "llama-ptuning"
port = 2001
url = "0.0.0.0:8812"

pd_cmd = "python3 api_client.py --url {0} --port {1} --model {2}".format(url, port, model)
print("pd_cmd: ", pd_cmd)
pd_process = subprocess.Popen(pd_cmd, shell=True, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, preexec_fn=os.setsid)

time.sleep( 5 )

# Modify OpenAI's API key and API base.
openai.api_key = "EMPTY"
openai.api_base = "http://0.0.0.0:"+str(port)+"/v1"


# Completion API
# 
stream = False

completion = openai.Completion.create(
    model=model,
    prompt="A robot may not injure a human being"
)

print("Completion results:")
print(completion)

# ChatCompletion API
# 

chat_completion = openai.ChatCompletion.create(
    model=model,
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }, {
        "role":
        "assistant",
        "content":
        "The Los Angeles Dodgers won the World Series in 2020."
    }, {
        "role": "user",
        "content": "Where was it played?"
    }])
print("Chat completion results:")
print(chat_completion)

os.killpg(os.getpgid(pd_process.pid), signal.SIGTERM) 