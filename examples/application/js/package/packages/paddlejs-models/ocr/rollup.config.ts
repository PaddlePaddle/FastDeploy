import path from 'path'
import { RollupOptions } from 'rollup'
import { string } from "rollup-plugin-string";
import rollupTypescript from 'rollup-plugin-typescript2'
import babel from 'rollup-plugin-babel'
import resolve from 'rollup-plugin-node-resolve'
import commonjs from 'rollup-plugin-commonjs'
import { eslint } from 'rollup-plugin-eslint'
import { DEFAULT_EXTENSIONS } from '@babel/core'

import pkg from './package.json'
import { paths } from "./build_package/util";


// rollup 配置项
const rollupConfig: RollupOptions = {
    input: paths.input,
    output: [
        // 输出 commonjs 规范的代码
        {
            file: path.join(paths.lib, 'index.js'),
            format: 'cjs',
            name: pkg.name,
        },
        // 输出 es 规范的代码
        {
            file: path.join(paths.lib, 'index.esm.js'),
            format: 'es',
            name: pkg.name,
        },
    ],
    external: ['@paddlejs-mediapipe/opencv',
        '@paddlejs/paddlejs-backend-webgl',
        '@paddlejs/paddlejs-core',
        '@types/node',
        'd3-polygon',
        'js-clipper',
        'number-precision'],
    // plugins 需要注意引用顺序
    plugins: [
        eslint({
            throwOnError: true,
            throwOnWarning: false,
            include: ['src/**/*.ts'],
            exclude: ['node_modules/**', 'lib/**', '*.js'],
        }),

        // 处理txt文件
        string({
            include: "src/ppocr_keys_v1.txt"
        }),
        // 使得 rollup 支持 commonjs 规范，识别 commonjs 规范的依赖
        commonjs(),

        // 配合 commnjs 解析第三方模块
        resolve({
            // 将自定义选项传递给解析插件
            customResolveOptions: {
                moduleDirectory: 'node_modules',
            },
        }),
        rollupTypescript(),
        babel({
            runtimeHelpers: true,
            // 只转换源代码，不运行外部依赖
            exclude: 'node_modules/**',
            // babel 默认不支持 ts 需要手动添加
            extensions: [
                ...DEFAULT_EXTENSIONS,
                '.ts',
            ],
        }),
    ],
}

export default rollupConfig
