[简体中文](../zh/online_serving/graceful_shutdown_service.md)

# Graceful Service Node Shutdown Solution

## 1. Core Objective
Achieve graceful shutdown of service nodes, ensuring no in-flight user requests are lost during service termination while maintaining overall cluster availability.

## 2. Solution Overview
This solution combines **Nginx reverse proxy**, **Gunicorn server**, **Uvicorn server**, and **FastAPI** working in collaboration to achieve the objective.

![graceful_shutdown](images/graceful_shutdown.png)

## 3. Component Introduction

### 1. Nginx: Traffic Entry Point and Load Balancer
- **Functions**:
  - Acts as a reverse proxy, receiving all external client requests and distributing them to upstream Gunicorn worker nodes according to load balancing policies.
  - Actively monitors backend node health status through health check mechanisms.
  - Enables instantaneous removal of problematic nodes from the service pool through configuration management, achieving traffic switching.

### 2. Gunicorn: WSGI HTTP Server (Process Manager)
- **Functions**:
  - Serves as the master process, managing multiple Uvicorn worker child processes.
  - Receives external signals (e.g., `SIGTERM`) and coordinates the graceful shutdown process for all child processes.
  - Daemonizes worker processes and automatically restarts them upon abnormal termination, ensuring service robustness.

### 3. Uvicorn: ASGI Server (Worker Process)
- **Functions**:
  - Functions as a Gunicorn-managed worker, actually handling HTTP requests.
  - Runs the FastAPI application instance, processing specific business logic.
  - Implements the ASGI protocol, supporting asynchronous request processing for high performance.

---

## Advantages

1. **Nginx**:
   - Can quickly isolate faulty nodes, ensuring overall service availability.
   - Allows configuration updates without downtime using `nginx -s reload`, making it transparent to users.

2. **Gunicorn** (Compared to Uvicorn's native multi-worker mode):
   - **Mature Process Management**: Built-in comprehensive process spawning, recycling, and management logic, eliminating the need for custom implementation.
   - **Process Daemon Capability**: The Gunicorn Master automatically forks new Workers if they crash, whereas in Uvicorn's `--workers` mode, any crashed process is not restarted and requires an external daemon.
   - **Rich Configuration**: Offers numerous parameters for adjusting timeouts, number of workers, restart policies, etc.

3. **Uvicorn**:
   - Extremely fast, built on uvloop and httptools.
   - Natively supports graceful shutdown: upon receiving a shutdown signal, it stops accepting new connections and waits for existing requests to complete before exiting.

---

## Graceful Shutdown Procedure

When a specific node needs to be taken offline, the steps are as follows:

1. **Nginx Monitors Node Health Status**:
   - Monitors the node's health status by periodically sending health check requests to it.

2. **Removal from Load Balancing**:
   - Modify the Nginx configuration to mark the target node as `down` and reload the Nginx configuration.
   - Subsequently, all new requests will no longer be sent to the target node.

3. **Gunicorn Server**:
   - Monitors for stop signals. Upon receiving a stop signal (e.g., `SIGTERM`), it relays this signal to all Uvicorn child processes.

4. **Sending the Stop Signal**:
   - Send a `SIGTERM` signal to the Uvicorn process on the target node, triggering Uvicorn's graceful shutdown process.

5. **Waiting for Request Processing**:
   - Wait for a period slightly longer than `timeout_graceful_shutdown` before forcefully terminating the service, allowing the node sufficient time to complete processing all received requests.

6. **Shutdown Completion**:
   - The node has now processed all remaining requests and exited safely.
