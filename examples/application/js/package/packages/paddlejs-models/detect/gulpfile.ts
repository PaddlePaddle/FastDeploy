import path from 'path'
import fse from 'fs-extra'
import { series } from "gulp"
import { paths, log } from "./build_package/util"
import rollupConfig from './rollup.config'
import { rollup } from 'rollup'
import {
  Extractor,
  ExtractorConfig,
  ExtractorResult,
} from '@microsoft/api-extractor'
/**
 * 这里是由于 'conventional-changelog' 未提供类型文件
 */
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import conventionalChangelog from 'conventional-changelog'

interface TaskFunc {
    // eslint-disable-next-line @typescript-eslint/ban-types
    (cb: Function): void
}

const CHANGE_TRACE = ['paddlejs-models/detect', 'paddle-js-models/detect', 'paddlejs-models', 'paddle-js-models', 'all']

/**
 * 删除 lib 文件
 * @param cb
 * @returns {Promise<void>}
 */
const clearLibFile: TaskFunc = async (cb) => {
    fse.removeSync(paths.lib)
    log.progress('Deleted lib file')
    cb()
}

/**
 * rollup 打包
 * @param cb
 */
const buildByRollup: TaskFunc = async (cb) => {
    const inputOptions = {
        input: rollupConfig.input,
        external: rollupConfig.external,
        plugins: rollupConfig.plugins,
    }
    const outOptions = rollupConfig.output
    const bundle = await rollup(inputOptions)

    // 写入需要遍历输出配置
    if (Array.isArray(outOptions)) {
        for (const outOption of outOptions) {
            await bundle.write(outOption)
        }
        cb()
        log.progress('Rollup built successfully')
    }
}

/**
 * api-extractor 整理 .d.ts 文件
 * @param cb
 */
const apiExtractorGenerate: TaskFunc = async (cb) => {
    const apiExtractorJsonPath: string = path.join(__dirname, './api-extractor.json')
    // 加载并解析 api-extractor.json 文件
    const extractorConfig: ExtractorConfig = await ExtractorConfig.loadFileAndPrepare(apiExtractorJsonPath)
    // 判断是否存在 index.d.ts 文件，这里必须异步先访问一边，不然后面找不到会报错
    const isdtxExist: boolean = await fse.pathExists(extractorConfig.mainEntryPointFilePath)
    // 判断是否存在 etc 目录，api-extractor需要该目录存在
    const isEtcExist: boolean = await fse.pathExists('./etc')

    if (!isdtxExist) {
        log.error('API Extractor not find index.d.ts')
        return
    }

    if (!isEtcExist) {
        fse.mkdirSync('etc');
        log.progress('Create folder etc for API Extractor')
    }

    // 调用 API
    const extractorResult: ExtractorResult = await Extractor.invoke(extractorConfig, {
        localBuild: true,
        // 在输出中显示信息
        showVerboseMessages: true,
    })

    if (extractorResult.succeeded) {
        // 删除多余的 .d.ts 文件
        const libFiles: string[] = await fse.readdir(paths.lib)
        for (const file of libFiles) {
            if (file.endsWith('.d.ts') && !file.includes('index')) {
                await fse.remove(path.join(paths.lib, file))
            }
        }
        log.progress('API Extractor completed successfully')
        // api-extractor 会生成 temp 文件夹，完成后进行删除
        fse.ensureDirSync('temp')
        fse.removeSync('temp')
        cb()
    } else {
        log.error(`API Extractor completed with ${extractorResult.errorCount} errors`
            + ` and ${extractorResult.warningCount} warnings`)
    }
}

/**
 * 完成
 * @param cb
 */
const complete: TaskFunc = (cb) => {
    log.progress('---- end ----')
    cb()
}

/**
 * 生成 CHANGELOG
 * @param cb
 */
export const changelog: TaskFunc = async (cb) => {
    const checkTrace = (chunk: string) => {
        for (const keyWord of CHANGE_TRACE) {
            if (chunk.includes(keyWord)) {
                return true
            }
        }
        return false
    }
    const changelogPath: string = path.join(paths.root, 'CHANGELOG.md')
    // 对命令 conventional-changelog -p angular -i CHANGELOG.md -w -r 0
    const changelogPipe = await conventionalChangelog({
        preset: 'angular',
        releaseCount: 0,
    })
    changelogPipe.setEncoding('utf8')

    const resultArray = ['# 更新日志\n\n']
    changelogPipe.on('data', (chunk) => {
        // 原来的 commits 路径是进入提交列表
        chunk = chunk.replace(/\/commits\//g, '/commit/')
        /**
         * title 或 指定跟踪 才会写入CHANGELOG
         */
        for (const log of chunk.split("\n")) {
            if (log.includes('# ') || log.includes('### ') || checkTrace(log)) {
                resultArray.push(log+"\n\n")
            }
        }
    })
    changelogPipe.on('end', async () => {
        fse.createWriteStream(changelogPath).write(resultArray.join(''))
        cb()
        log.progress('CHANGELOG generation completed')
    })
}


exports.build = series(clearLibFile, buildByRollup, apiExtractorGenerate, complete)
