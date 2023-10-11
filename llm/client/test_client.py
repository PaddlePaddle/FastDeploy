from Client import grpcClient

client  = grpcClient(base_url="0.0.0.0:8812",
                     model_name="llama-ptuning",
                     timeout= 100)
result = client.generate("Hello, how are you")
print(result)