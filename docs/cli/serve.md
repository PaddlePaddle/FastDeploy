# serve: API Service Deployment

`fastdeploy serve` provides service deployment compatible with the OpenAI API protocol.

## Parameters

The following table lists the available options:

| Option     | Description                                              | Default |
| ---------- | -------------------------------------------------------- | ------- |
| `--config` | Read CLI options from a configuration file (YAML format) | None    |

For more parameter details, see: [FastDeploy Parameter Documentation](../parameters.md)

## Examples

```bash
# Start the FastDeploy API server
fastdeploy serve --model baidu/ERNIE-4.5-0.3B-Paddle

# Start the server with a specified port
fastdeploy serve --model baidu/ERNIE-4.5-0.3B-Paddle --port 8000
```
