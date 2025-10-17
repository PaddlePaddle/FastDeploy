[English](../../usage/fastdeploy_unit_test_guide.md)

# Fastdeploy 单测规范
1. 测试命名规范
   - 测试文件以 test_ 开头；
   - 测试方法命名建议采用 test_func_behavior 格式，或 test_func，保持语义清晰。

2. 目录结构

   - 所有单测应统一放在 test/ 目录，并根据 FastDeploy 模块结构划分子目录，便于维护和定位。可参考[vllm](https://github.com/vllm-project/vllm/tree/main/tests)

3. 覆盖范围，每个关键模块/类/函数需覆盖

   - 正常路径 （标准输入，主流程）
   - 边界输入 （空值、极大/极小值、边界条件）
   - 异常输入 （非法参数、错误格式等，需校验异常行为是否符合预期，例如exception的内容）

4. Case编写与执行

   - 测试用例应支持在 CI 、本地环境中通过 pytest 或 unittest 一键运行
   - 使用明确的 assert 判断模块行为或返回值，避免仅打印输出
   - 每个测试用例应具备良好的原子性，每个用例聚焦一个行为，避免多个功能混测
   - 测试之间保持独立，不依赖运行顺序、不共享全局状态，即用例之间必须是解耦的
   - 自定义算子需要有C++级别的单测或者是基于Paddle单层组网的前反向测试 ，参考晓光的文档 开发套件自定义算子和框架非正式算子开发规范(试运行)

5. WebServer 相关测试

   - 除非必要，请避免在单测中直接起 WebServer 进行端到端测试
   - 推荐使用 mock 替代网络请求、模块上下游调用等，以实现更稳定、可控的模块级原子性测试
   - 如果确需通过 HTTP 发起请求
   - QA 会提供一套端口注入规范（将端口写入环境变量，Case可直接加载变量内容）（预计下周三完成规范和脚本编写）类似下图
   - 测试用例中读取环境变量，避免硬编码端口，保障 CI 多实例并发下可复用，且便于本地调试

<img width="500" height="240" alt="Image" src="https://github.com/user-attachments/assets/9bc052d0-0aa0-4e4f-97ce-b8b920f20203" />
