# chat: Conversational Generation

`fastdeploy chat` interacts with a running API server to generate chat responses.

## Parameters

| Parameter       | Description                                                                                          | Default                                              |
| --------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| --url           | URL of the running OpenAI-compatible RESTful API server                                              | [http://localhost:8000/v1](http://localhost:8000/v1) |
| --model-name    | Name of the model to use for prompt completion; defaults to the first model listed in the models API | None                                                 |
| --api-key       | API key for OpenAI services; overrides environment variable if provided                              | None                                                 |
| --system-prompt | Specifies the system prompt used in the chat template                                                | None                                                 |
| -q, --quick     | Sends a single prompt as a MESSAGE, prints the response, and exits                                   | None                                                 |

## Examples

```
# Connect directly to a local API
fastdeploy chat

# Specify an API URL
fastdeploy chat --url http://{fastdeploy-serve-host}:{fastdeploy-serve-port}/v1

# Send a single quick prompt
fastdeploy chat --quick "hi"
```
