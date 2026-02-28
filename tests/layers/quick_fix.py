#!/usr/bin/env python3
# 快速修正测试文件中的函数名


# 读取文件
with open("test_dsmla_writecache_integration.py", "r", encoding="utf-8") as f:
    content = f.read()

# 将 ds_mla_write_cache 替换为 dsmla_write_cache
content = content.replace("ds_mla_write_cache", "dsmla_write_cache")
content = content.replace("_ds_mla_", "_dsmla_")

# 写回文件
with open("test_dsmla_writecache_integration.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✓ 修正完成：将 ds_mla_write_cache 替换为 dsmla_write_cache")
