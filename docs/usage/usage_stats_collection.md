[简体中文](../zh/usage/usage_stats_collection.md)

# usage collection

Fastdeploy collects anonymous usage data by default to help the engineering team understand common hardware and model configurations, thereby prioritizing optimizations for frequently used configurations. All data collected is transparent and contains no sensitive information.

# Data Collected

The data inventory collected by the latest version of Fastdeploy can be found here：[usage_lib.py](../../fastdeploy/usage/usage_lib.py)<br>
You can preview the collected data by running the following command：<br>
`tail ~/.config/fastdeploy/usage_stats.json`

# Opting Out

You can opt out of usage collection by setting the `DO_NOT_TRACK` environment variable：<br>
`export DO_NOT_TRACK=1`
