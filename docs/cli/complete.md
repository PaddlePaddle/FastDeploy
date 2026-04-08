# complete: Text Completion Generation

`fastdeploy complete` generates text completions based on a given prompt.

## Parameters

| Parameter       | Description                                                        | Default                                              |
| --------------- | ------------------------------------------------------------------ | ---------------------------------------------------- |
| --url           | URL of the running OpenAI-compatible RESTful API server            | [http://localhost:8000/v1](http://localhost:8000/v1) |
| --model-name    | Name of the model used for prompt completion                       | None                                                 |
| --api-key       | API key for OpenAI services                                        | None                                                 |
| --system-prompt | Specifies the system prompt used in the chat template              | None                                                 |
| -q, --quick     | Sends a single prompt as a MESSAGE, prints the response, and exits | None                                                 |

## Examples

```
# Connect directly to a local API
fastdeploy complete

# Specify an API URL
fastdeploy complete --url http://{fastdeploy-serve-host}:{fastdeploy-serve-port}/v1

# Generate a quick completion
fastdeploy complete --quick "The future of AI is"
```
