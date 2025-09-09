1. 服务可以支持多大并发？
- 服务部署时推荐配置环境变量export ENABLE_V1_KVCACHE_SCHEDULER=1
- 服务在启动时需要配置```max-num-seqs```，此参数用于表示Decode阶段的最大Batch数，如果并发超过此值，则超出的请求会排队等待处理, 常规情况下你可以将```max-num-seqs```配置为128，保持在较高的范围，实际并发由发压客户端来决定。
- ```max-num-seqs```仅表示设定的上限，但实际上服务能并发处理的上限取决于KVCache的大小，在启动服务后，查看log/worker_process.log会看到类似```num_blocks_global: 17131```的日志，这表明当前服务的KVCache Block数量为17131, 17131*block_size(默认64）即知道总共可缓存的Token数量，例如此处为17131*64=1096384。如果你的请求数据平均输入和输出Token之和为20K，那么服务实际可以处理的并发大概为1096384/20k=53
