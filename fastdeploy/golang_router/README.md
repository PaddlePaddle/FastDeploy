# fd-router

A high-performance Go routing framework providing flexible request routing, middleware support, and health check functionality.

## Features

- High-performance HTTP/HTTPS server
- RESTful API routing support
- Extensible middleware system
- Dynamic configuration management
- Built-in health check and monitoring
- Request rate limiting and load balancing
- Detailed logging and metrics collection

## Quick Start

### Prerequisites

- Go 1.21 or higher

### Compilation

```bash
./build.sh
```

### Configuration

1. Copy and modify the configuration template:

```bash
cp config/config.example.yaml config/config.yaml
```

2. Main configuration items:

```yaml
server:
  port: "8080" # Listening port
  host: "0.0.0.0" # Listening address
  mode: "debug" # Startup mode: debug, release, test
  splitwise: true # true enables pd separation mode, false enables non-pd separation mode

scheduler:
  policy: "request_num" # Scheduling policy (optional): random, power_of_two, round_robin, process_tokens, request_num
  prefill-policy: "process_tokens" # Prefill node scheduling policy in pd separation mode
  decode-policy: "request_num" # Decode node scheduling policy in pd separation mode
  interval-cleanup-secs: 60 # Cache cleanup interval for cache-aware strategy

manager:
  health-failure-threshold: 3 # Health check failure threshold, node considered unhealthy if exceeded
  health-success-threshold: 2 # Health check success threshold, node considered healthy if exceeded
  health-check-timeout-secs: 5 # Health check timeout time
  health-check-interval-secs: 5 # Health check interval time
  health-check-endpoint: /health # Health check endpoint

log:
  level: "info"  # Log level
  output: "file" # Log output method: stdout, file
```

Refer to examples/run_with_config to start router using configuration file

### Running

```bash
go run cmd/main.go

# Or run using binary
./run.sh
```

## Project Structure

```
.
├── cmd/              # Main entry point
├── config/           # Configuration files
├── internal/         # Core implementation code
│   ├── common/       # Common interface definitions
│   ├── config/       # Configuration handling
│   ├── gateway/      # API gateway implementation
│   ├── manager/      # Route management
│   ├── middleware/   # Middleware implementation
│   ├── router/       # Core routing logic
│   └── scheduler/    # Scheduler implementation
├── logs/             # Log directory
├── output/           # Build output
├── pkg/              # Reusable components
│   ├── logger/       # Logging component
│   └── metrics/      # Monitoring metrics
├── build.sh          # Build script
├── go.mod            # Go module definition
├── go.sum            # Dependency checksum
├── Makefile          # Build management
├── README.md         # Project documentation (Chinese)
└── run.sh            # Startup script
```

### Running Tests

```bash
make test
```

## Contributing

Issues and pull requests are welcome!