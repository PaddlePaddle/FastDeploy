/**
 * @file detect model
 */

import { Runner } from '@paddlejs/paddlejs-core';
import '@paddlejs/paddlejs-backend-webgl';

let detectRunner = null as Runner;

export async function init() {
    detectRunner = new Runner({
        modelPath: 'https://paddlejs.bj.bcebos.com/models/fuse/detect/detect_fuse_activation/model.json',
        fill: '#fff',
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        bgr: true,
        keepRatio: false,
        webglFeedProcess: true
    });

    await detectRunner.init();
}

export async function detect(image) {
    const output = await detectRunner.predict(image);
    // 阈值
    const thresh = 0.3;
    return output.filter(item => item[1] > thresh);
}
