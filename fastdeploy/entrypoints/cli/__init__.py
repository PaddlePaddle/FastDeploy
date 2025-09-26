from fastdeploy.entrypoints.cli.benchmark.latency import BenchmarkLatencySubcommand
from fastdeploy.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand

__all__: list[str] = [
    "BenchmarkLatencySubcommand",
    "BenchmarkServingSubcommand",
]
