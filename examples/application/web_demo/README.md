# 工程化打包示例

该项目基于 ocr 源码实现工程化打包样例。主要涉及以下技术点：
1. gulp 流水线
2. rollup ts打包
3. eslint 代码格式维护
4. typescript
5. api-extractor API提取；生成d.ts文件
6. jest（暂定，可以引入其他e2e） 测试
7. conventional-changelog-cli 生成changelog
8. husky 代码上传维护
9. yalc 本地打包测试
10. pnpm 支持monorepo

## 实现效果

1. 使用 rollup 一次性打包生成 commonjs 和 es 规范的代码；同时具有可扩展性；目前由于依赖的cv库有些问题；就没有配置umd打包。
2. 打包时基于 api-extractor 实现 d.ts 文件生成，实现支持 ts 引入生成我们的包
3. 基于 jest 支持测试并显示测试相关覆盖率等
4. 基于 ts 和 eslint 维护代码风格，保证代码更好开发
5. 基于 conventional-changelog-cli 实现自定义关键词生成对应生成changelog
6. 基于 husky 实现代码上传前代码格式验证以及commit格式校验
7. 基于 yalc 实现本地打包开发测试
8. publish 前可以自动进行代码风格校验和修复，代码测试，代码打包，changelog生成
9. pnpm支持monorepo

## 开发

package管理全部使用pnpm；同时需要全局安装yalc。[yalc参考资料](https://zhuanlan.zhihu.com/p/469010320)
