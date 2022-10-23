import { Runner } from '@paddlejs/paddlejs-core';
import '@paddlejs/paddlejs-backend-webgl';

interface FeedShape {
    fw: number;
    fh: number;
}
export interface ModelConfig {
    modelPath?: string;
    mean?: number[];
    std?: number[];
    feedShape?: FeedShape;
}
interface NaturalSize {
    naturalWidth: number;
    naturalHeight: number;
}
export interface TransformData {
    left: number;
    width: number;
    top: number;
    height: number;
    confidence: number;
}

// 默认模型数据
const defaultFeedShape: FeedShape = {
    fw: 1024,
    fh: 1024
};
const defaultModelConfig: ModelConfig = {
    modelPath: 'https://paddlejs.cdn.bcebos.com/models/fuse/facedetect_opt/model.json',
    mean: [0.407843137, 0.458823529, 0.482352941],
    std: [0.5, 0.5, 0.5]
};

export class FaceDetector {
    modelConfig: ModelConfig = defaultModelConfig;
    feedShape: FeedShape = defaultFeedShape;
    runner: Runner = null;
    inputSize = {} as NaturalSize;
    scale: number | undefined;

    constructor(modeConfig?: ModelConfig) {
        this.modelConfig = Object.assign(this.modelConfig, modeConfig);
        this.feedShape = Object.assign(this.feedShape, modeConfig?.feedShape);
    }

    async init() {
        this.runner = new Runner(this.modelConfig);
        await this.runner.init();
    }

    async detect(
        input: HTMLImageElement,
        opts?
    ) {
        this.inputSize.naturalWidth = input.naturalWidth;
        this.inputSize.naturalHeight = input.naturalHeight;
        const { shrink = 0.4, threshold = 0.6 } = Object.assign({}, opts);

        const inputFeed = this.preprocess(input, shrink);

        // 预测
        const dataOut = await this.runner.predictWithFeed(inputFeed);
        return this.postprocessor(dataOut, threshold);
    }

    preprocess(input: HTMLImageElement, shrink: number) {
        // shrink --> scale
        const scale = this.scale = Math.min(this.inputSize.naturalWidth, this.inputSize.naturalHeight) * shrink;
        return this.runner.mediaProcessor.process(
            input,
            Object.assign({}, this.modelConfig, { scale }),
            this.feedShape
        );
    }

    postprocessor(dataOut, threshold: number) {
        // data filter 筛掉小于阈值项
        const dataFilt = dataOut.filter(item => item[1] && item[1] > threshold);
        // data transform
        return dataFilt.map(item => this.transformData(item));
    }

    transformData(data: Array<number>) {
        const transformRes = {} as TransformData;
        const { fw, fh } = this.feedShape;
        const { naturalWidth, naturalHeight } = this.inputSize;
        let dWidth;
        let dHeight;

        if (naturalWidth > naturalHeight) {
            dHeight = this.scale;
            dWidth = dHeight * naturalWidth / naturalHeight;
        }
        else {
            dWidth = this.scale;
            dHeight = dWidth * naturalHeight / naturalWidth;
        }
        const dx = (fw - dWidth) / 2;
        const dy = (fh - dHeight) / 2;
        transformRes.confidence = data[1];
        transformRes.left = (data[2] * fw - dx) / dWidth;
        transformRes.width = (data[4] - data[2]) * fw / dWidth;
        transformRes.top = (data[3] * fh - dy) / dHeight;
        transformRes.height = (data[5] - data[3]) * fh / dHeight;
        return transformRes;
    }
}

export function createImage(imgPath: string): Promise<HTMLImageElement> {
    return new Promise(resolve => {
        const image = new Image();
        image.crossOrigin = 'anonymous';
        image.onload = () => {
            resolve(image);
        };
        image.src = imgPath;
    });
}
