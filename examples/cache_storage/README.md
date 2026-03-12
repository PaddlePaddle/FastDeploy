# Global Cache Pooling Examples

This directory contains example scripts for Global Cache Pooling with MooncakeStore.

## Documentation

- [English Documentation](../../docs/features/global_cache_pooling.md)
- [中文文档](../../docs/zh/features/global_cache_pooling.md)

## Quick Start

```bash
# Multi-instance scenario
bash run.sh

# PD disaggregation scenario
bash run_03b_pd_storage.sh
```

## Scripts

| Script | Scenario | Description |
|--------|----------|-------------|
| `run.sh` | Multi-Instance | Two standalone instances sharing cache |
| `run_03b_pd_storage.sh` | PD Disaggregation | P+D instances with global cache pooling |

## Files

- `mooncake_config.json` - Mooncake configuration file
- `utils.sh` - Utility functions for scripts
- `stop.sh` - Stop all running services
