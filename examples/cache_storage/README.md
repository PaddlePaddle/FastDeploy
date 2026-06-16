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

# High-availability scenario (multi-master + failover)
bash run_ha.sh         # etcd backend
bash run_ha_redis.sh   # redis backend
```

## Scripts

| Script | Scenario | Description |
|--------|----------|-------------|
| `run.sh` | Multi-Instance | Two standalone instances sharing cache |
| `run_03b_pd_storage.sh` | PD Disaggregation | P+D instances with global cache pooling |
| `run_ha.sh` | High Availability (etcd) | Self-contained: starts etcd + 3 masters with leader election, then kills the leader and re-verifies pooling with a fresh prompt after re-election |
| `run_ha_redis.sh` | High Availability (redis) | Same flow as `run_ha.sh`, but uses a single redis instead of etcd for leader election |

## Files

- `mooncake_config.json` - Mooncake configuration file (single master)
- `ha_mooncake_config.json` - Mooncake HA client config (etcd-based master discovery)
- `ha_redis_mooncake_config.json` - Mooncake HA client config (redis-based master discovery)
- `utils.sh` - Utility functions for scripts
- `stop.sh` - Stop all running services
