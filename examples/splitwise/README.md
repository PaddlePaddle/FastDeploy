
PD分离式部署，请参考[使用文档](../../docs/zh/features/disaggregated.md)。

PD分离式部署，推荐使用Router来做请求调度（即是V1模式）。

启动脚本：

* `start_v1_tp1.sh`：使用Router调度，P和D实例是TP1。
* `start_v1_tp2.sh`：使用Router调度，P和D实例是TP2。
* `start_v1_dp2.sh`：使用Router调度，P和D实例是DP2 TP1。
