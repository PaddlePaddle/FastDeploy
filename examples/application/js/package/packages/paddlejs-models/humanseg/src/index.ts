/**
 * @file humanseg model
 */

import { Runner, env } from '@paddlejs/paddlejs-core';
import '@paddlejs/paddlejs-backend-webgl';
import WebGLImageFilter from '../thirdParty/webgl-image-filter';

let runner = null as Runner;
let inputElement: any = null;

let WIDTH = 398;
let HEIGHT = 224;

let backgroundSize: any = null;

const blurFilter = new WebGLImageFilter();
blurFilter.reset();
blurFilter.addFilter('blur', 10);


export async function load(needPreheat = true, enableLightModel = false, customModel = null) {
    const modelpath = 'https://paddlejs.bj.bcebos.com/models/fuse/humanseg/humanseg_398x224_fuse_activation/model.json';
    const lightModelPath = 'https://paddlejs.bj.bcebos.com/models/fuse/humanseg/humanseg_288x160_fuse_activation/model.json';
    const path = customModel
        ? customModel
        : enableLightModel ? lightModelPath : modelpath;
    if (enableLightModel) {
        WIDTH = 288;
        HEIGHT = 160;
    }

    runner = new Runner({
        modelPath: path,
        needPreheat: needPreheat !== undefined ? needPreheat : true,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        webglFeedProcess: true
    });

    env.set('webgl_pack_channel', true);
    env.set('webgl_pack_output', true);

    await runner.init();

    if (runner.feedShape) {
        WIDTH = runner.feedShape.fw;
        HEIGHT = runner.feedShape.fh;
    }
}

export async function preheat() {
    return await runner.preheat();
}

export async function getGrayValue(input: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement) {
    inputElement = input;
    const seg_values = await runner.predict(input);
    backgroundSize = genBackgroundSize();
    return {
        width: WIDTH,
        height: HEIGHT,
        data: seg_values
    };
}


function genBackgroundSize() {
    // 缩放后的宽高
    let sw = WIDTH;
    let sh = HEIGHT;
    const ratio = sw / sh;
    const inputWidth = inputElement.naturalWidth || inputElement.width;
    const inputHeight = inputElement.naturalHeight || inputElement.height;
    let x = 0;
    let y = 0;
    let bx = 0;
    let by = 0;
    let bh = inputHeight;
    let bw = inputWidth;
    const origin_ratio = inputWidth / inputHeight;
    // target的长宽比大些 就把原图的高变成target那么高
    if (ratio / origin_ratio >= 1) {
        sw = sh * origin_ratio;
        x = Math.floor((WIDTH - sw) / 2);
        bw = bh * ratio;
        bx = Math.floor((bw - inputWidth) / 2);
    }
    // target的长宽比小些 就把原图的宽变成target那么宽
    else {
        sh = sw / origin_ratio;
        y = Math.floor((HEIGHT - sh) / 2);
        bh = bw / ratio;
        by = Math.floor((bh - inputHeight) / 2);
    }
    return {
        x,
        y,
        sw,
        sh,
        bx,
        by,
        bw,
        bh
    };
}


/**
 * draw human seg
 * @param {Array} seg_values seg values of the input image
 * @param {HTMLCanvasElement} canvas the dest canvas draws the pixels
 * @param {HTMLCanvasElement} backgroundCanvas the background canvas draws the pixels
 */
export function drawHumanSeg(
    seg_values: number[],
    canvas: HTMLCanvasElement,
    backgroundCanvas?: HTMLCanvasElement | HTMLImageElement
) {
    const inputWidth = inputElement.naturalWidth || inputElement.width;
    const inputHeight = inputElement.naturalHeight || inputElement.height;

    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    canvas.width = WIDTH;
    canvas.height = HEIGHT;

    const tempCanvas = document.createElement('canvas') as HTMLCanvasElement;
    const tempContext = tempCanvas.getContext('2d') as CanvasRenderingContext2D;
    tempCanvas.width = WIDTH;
    tempCanvas.height = HEIGHT;

    const tempScaleData = ctx.getImageData(0, 0, WIDTH, HEIGHT);
    tempContext.drawImage(inputElement, backgroundSize.x, backgroundSize.y, backgroundSize.sw, backgroundSize.sh);
    const originImageData = tempContext.getImageData(0, 0, WIDTH, HEIGHT);
    for (let i = 0; i < WIDTH * HEIGHT; i++) {
        if (seg_values[i + WIDTH * HEIGHT] * 255 > 100) {
            tempScaleData.data[i * 4] = originImageData.data[i * 4];
            tempScaleData.data[i * 4 + 1] = originImageData.data[i * 4 + 1];
            tempScaleData.data[i * 4 + 2] = originImageData.data[i * 4 + 2];
            tempScaleData.data[i * 4 + 3] = seg_values[i + WIDTH * HEIGHT] * 255;
        }
    }

    tempContext.putImageData(tempScaleData, 0, 0);
    canvas.width = inputWidth;
    canvas.height = inputHeight;
    backgroundCanvas
    && ctx.drawImage(backgroundCanvas, -backgroundSize.bx, -backgroundSize.by, backgroundSize.bw, backgroundSize.bh);
    ctx.drawImage(tempCanvas, -backgroundSize.bx, -backgroundSize.by, backgroundSize.bw, backgroundSize.bh);
}

/**
 * draw human seg
 * @param {HTMLCanvasElement} canvas the dest canvas draws the pixels
 * @param {Array} seg_values seg_values of the input image
 */
export function blurBackground(seg_values: number[], dest_canvas) {
    const inputWidth = inputElement.naturalWidth || inputElement.width;
    const inputHeight = inputElement.naturalHeight || inputElement.height;
    const tempCanvas = document.createElement('canvas') as HTMLCanvasElement;
    const tempContext = tempCanvas.getContext('2d') as CanvasRenderingContext2D;
    tempCanvas.width = WIDTH;
    tempCanvas.height = HEIGHT;

    const dest_ctx = dest_canvas.getContext('2d') as CanvasRenderingContext2D;
    dest_canvas.width = inputWidth;
    dest_canvas.height = inputHeight;

    const tempScaleData = tempContext.getImageData(0, 0, WIDTH, HEIGHT);
    tempContext.drawImage(inputElement, backgroundSize.x, backgroundSize.y, backgroundSize.sw, backgroundSize.sh);
    const originImageData = tempContext.getImageData(0, 0, WIDTH, HEIGHT);

    blurFilter.dispose();
    const blurCanvas = blurFilter.apply(tempCanvas);

    for (let i = 0; i < WIDTH * HEIGHT; i++) {
        if (seg_values[i + WIDTH * HEIGHT] * 255 > 150) {
            tempScaleData.data[i * 4] = originImageData.data[i * 4];
            tempScaleData.data[i * 4 + 1] = originImageData.data[i * 4 + 1];
            tempScaleData.data[i * 4 + 2] = originImageData.data[i * 4 + 2];
            tempScaleData.data[i * 4 + 3] = seg_values[i + WIDTH * HEIGHT] * 255;
        }
    }

    tempContext.putImageData(tempScaleData, 0, 0);

    dest_ctx.drawImage(blurCanvas, -backgroundSize.bx, -backgroundSize.by, backgroundSize.bw, backgroundSize.bh);
    dest_ctx.drawImage(tempCanvas, -backgroundSize.bx, -backgroundSize.by, backgroundSize.bw, backgroundSize.bh);
}


/**
 * draw mask without human
 * @param {Array} seg_values seg_values of the input image
 * @param {HTMLCanvasElement} dest the dest canvas draws the pixels
 * @param {HTMLCanvasElement} canvas background canvas
 */
export function drawMask(seg_values: number[], dest: HTMLCanvasElement, canvas: HTMLCanvasElement) {
    const tempCanvas = document.createElement('canvas') as HTMLCanvasElement;
    const tempContext = tempCanvas.getContext('2d') as CanvasRenderingContext2D;
    tempCanvas.width = WIDTH;
    tempCanvas.height = HEIGHT;
    tempContext.drawImage(canvas, 0, 0, WIDTH, HEIGHT);
    const dest_ctx = dest.getContext('2d') as CanvasRenderingContext2D;
    dest.width = WIDTH;
    dest.height = HEIGHT;

    const tempScaleData = tempContext.getImageData(0, 0, WIDTH, HEIGHT);
    for (let i = 0; i < WIDTH * HEIGHT; i++) {
        if (seg_values[i + WIDTH * HEIGHT] * 255 > 150) {
            tempScaleData.data[i * 4 + 3] = seg_values[i] * 255;
        }
    }

    tempContext.putImageData(tempScaleData, 0, 0);
    dest_ctx.drawImage(tempCanvas, 0, 0, WIDTH, HEIGHT);
}
