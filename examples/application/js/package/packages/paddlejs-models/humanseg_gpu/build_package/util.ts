import path from "path";
import chalk from "chalk";

export const paths = {
    root: path.join(__dirname, '../'),
    input: path.join(__dirname, '../src/index_gpu.ts'),
    lib: path.join(__dirname, '../lib'),
}

export const log = {
    progress: (text: string) => {
        console.log(chalk.green(text))
    },
    error: (text: string) => {
        console.log(chalk.red(text))
    },
}
