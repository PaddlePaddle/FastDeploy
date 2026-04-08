## Observability Example Configuration (`examples/observability`)

This directory provides a complete, Docker Compose–based observability example environment, including:

* **Prometheus**: Metrics collection
* **Grafana**: Metrics visualization
* **OpenTelemetry Collector**: Distributed tracing data ingestion and processing

Developers can use this example to **launch a local monitoring and tracing system with a single command**.

---

### Prerequisites

Please make sure the following components are installed in advance:

* Docker
* Docker Compose (or a newer Docker CLI version that supports `docker compose`)

---

### Usage

#### Start All Services

Enter the directory:

```bash
cd examples/observability
```

Run the following command to start the complete monitoring and tracing stack:

```bash
docker compose -f docker-compose.yaml up -d
```

After startup, you can access:

* **Prometheus**: [http://localhost:9090](http://localhost:9090)
* **Grafana**: [http://localhost:3000](http://localhost:3000)
* **OTLP receiver**: Applications should send traces to the default ports of the OTel Collector (usually `4317` or `4318`)

  * gRPC: `4317`
  * HTTP: `4318`
* **Jaeger UI**: [http://localhost:16886](http://localhost:16886)

**Notes:**

* Update the Prometheus scrape targets to match your actual application endpoints.
* Map Grafana’s service port to a port that is accessible on your machine.
* Map the Jaeger UI port to a port that is accessible on your machine.
* When starting the full stack, there is no need to start individual sub-services separately.

---

#### Start Metrics Services Only

Enter the directory:

```bash
cd examples/observability/metrics
```

Run the following command:

```bash
docker compose -f prometheus_compose.yaml up -d
```

After startup, you can access:

* **Grafana**: [http://localhost:3000](http://localhost:3000)

---

#### Start Tracing Services Only

Enter the directory:

```bash
cd examples/observability/tracing
```

Run the following command:

```bash
docker compose -f tracing_compose.yaml up -d
```

After startup, you can access:

* **OTLP receiver**: Applications should send traces to the default ports of the OTel Collector (usually `4317` or `4318`)

  * gRPC: `4317`
  * HTTP: `4318`
* **Jaeger UI**: [http://localhost:16886](http://localhost:16886)

---

### Directory Structure and File Descriptions

#### Core Startup File

| File Name             | Purpose    | Description                                                                                                                                                         |
| --------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docker-compose.yaml` | Main entry | Defines and starts the full observability stack (Prometheus, Grafana, OTel Collector, and Jaeger). This is the single entry point to launch the entire environment. |

---

#### Metrics and Monitoring Configuration

| File / Directory                                    | Purpose                  | Description                                                                                                               |
| --------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `metrics`                                           | Metrics root directory   | Contains all Prometheus- and metrics-related configurations.                                                              |
| `prometheus.yaml`                                   | Prometheus main config   | Defines scrape targets, global scrape parameters, and optional recording rules. All monitored endpoints are defined here. |
| `prometheus_compose.yaml`                           | Prometheus Docker config | Defines the Prometheus container, volume mounts, and network settings.                                                    |
| `grafana/datasources/datasource.yaml`               | Datasource configuration | Configures how Grafana connects to Prometheus.                                                                            |
| `grafana/dashboards/config/dashboard.yaml`          | Dashboard provisioning   | Specifies the locations of dashboard JSON files to be loaded.                                                             |
| `grafana/dashboards/json/fastdeploy-dashboard.json` | Dashboard definition     | Contains visualization layouts and queries for `fastdeploy` monitoring metrics.                                           |

---

#### Distributed Tracing Configuration

| File / Directory                                                                | Purpose                | Description                                                            |
| ------------------------------------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------- |
| `tracing`                                                                       | Tracing root directory | Contains all configurations related to distributed tracing.            |
| `opentelemetry.yaml`                                                            | OTel Collector config  | Defines the Collector data pipelines:                                  |
| • **receivers**: receive OTLP data (traces, metrics, logs)                      |                        |                                                                        |
| • **processors**: data processing and batching                                  |                        |                                                                        |
| • **exporters**: export data to tracing backends (such as Jaeger) or files      |                        |                                                                        |
| • **extensions**: health check, pprof, and zpages                               |                        |                                                                        |
| • **pipelines**: define complete processing flows for traces, metrics, and logs |                        |                                                                        |
| `tracing_compose.yaml`                                                          | Tracing Docker config  | Defines the container configuration for the OTel Collector and Jaeger. |

---

### Customization

#### 4.1 Modify Metrics Scrape Targets

If your application’s metrics endpoint, port, or path changes, edit:

```plain
metrics/prometheus.yaml
```

---

#### 4.2 Adjust Tracing Sampling Rate or Processing Logic

Edit:

```plain
tracing/opentelemetry.yaml
```

---

#### 4.3 Add Custom Grafana Dashboards

1. Add the new dashboard JSON file to:

```plain
grafana/dashboards/json/
```

2. Register the dashboard so Grafana can load it automatically by editing:

```plain
grafana/dashboards/config/dashboard.yaml
```
