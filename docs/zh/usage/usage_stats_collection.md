[English](../../usage/usage_stats_collection.md)

# 使用信息收集

Fastdeploy默认会收集匿名使用数据，旨在帮助工程团队了解常见的硬件与模型配置，从而优先优化使用频率较高的配置。所有收集的数据均公开透明，不涉及任何敏感信息。

# 收集内容

最新版本Fastdeploy所收集的数据清单可在此处查看：[usage_lib.py](../../../fastdeploy/usage/usage_lib.py)<br>
您可以通过运行以下命令预览所收集的数据：<br>
`tail ~/.config/fastdeploy/usage_stats.json`

# 关闭使用信息收集

您可以通过设置 `DO_NOT_TRACK` 环境变量，来退出使用情况统计信息的收集：<br>
`export DO_NOT_TRACK=1`
