/**
 * @file mobilenet model
 */

import { Runner, env } from '@paddlejs/paddlejs-core';
import '@paddlejs/paddlejs-backend-webgl';

interface ModelConfig {
    path: string;
    mean?: number[];
    std?: number[];
    needPreheat?: boolean;
}

interface MobilenetMap {
    [key: string]: string
}

let mobilenetMap = null as any;
let runner = null as Runner;

export async function load(config: ModelConfig, map: string[] | MobilenetMap) {
    mobilenetMap = map;

    const {
        path,
        mean,
        std,
        needPreheat = true
    } = config;

    runner = new Runner({
        modelPath: path,
        fill: '#fff',
        mean: mean || [],
        std: std || [],
        scale: 256,
        needPreheat
    });
    env.set('webgl_pack_channel', true);
    await runner.init();
}

// 获取数组中的最大值索引
function getMaxItem(datas: number[] = []) {
    const max: number = Math.max.apply(null, datas);
    const index: number = datas.indexOf(max);
    return index;
}


export async function classify(image) {
    const res = await runner.predict(image);
    const maxItem = getMaxItem(res);
    const result = mobilenetMap[`${maxItem}`];
    return result;
}

