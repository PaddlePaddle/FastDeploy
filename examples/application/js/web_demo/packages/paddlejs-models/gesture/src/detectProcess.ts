import WarpAffine from './warpAffine';

class DetectProcess {
    private modelResult: any;
    private output_size: number;
    private anchor_num: number;
    private detectResult: any;
    private hasHand: number;
    private originCanvas: HTMLCanvasElement;
    private originImageData: ImageData;
    private maxIndex!: number;
    private box: any;
    private feed: any;
    private anchors: any;
    private mtr: any;
    private source!: any[][];
    private imageData!: ImageData;
    private pxmin: any;
    private pxmax: any;
    private pymin: any;
    private pymax: any;
    private kp!: number[][];
    private tw!: number;
    private th!: number;
    private triangle: any;
    private results: any;

    constructor(result, canvas) {
        this.modelResult = result;
        this.output_size = 10;
        this.anchor_num = 1920;
        this.detectResult = new Array(14).fill('').map(() => []);
        this.hasHand = 0;
        this.originCanvas = canvas;
        const ctx = this.originCanvas.getContext('2d');
        this.originImageData = ctx!.getImageData(0, 0, 256, 256);
    }

    async outputBox(results) {
        this.getMaxIndex();
        if (this.maxIndex === -1) {
            this.hasHand = 0;
            return false;
        }
        this.hasHand = 1;
        await this.splitAnchors(results);
        this.decodeAnchor();
        // 求关键点
        this.decodeKp();
        // 求手势框
        this.decodeBox();
        return this.box;
    }

    async outputFeed(paddle) {
        this.decodeTriangle();
        this.decodeSource();
        this.outputResult();
        // 提取仿射变化矩阵
        this.getaffinetransform();
        // 对图片进行仿射变换
        await this.warpAffine();
        this.allReshapeToRGB(paddle);
        return this.feed;
    }

    async splitAnchors(results) {
        this.results = results;
        const anchors: number[][] = new Array(this.anchor_num).fill('').map(() => []);
        const anchor_num = this.anchor_num;
        for (let i = 0; i < anchor_num; i++) {
            const tmp0 = results[i * 4];
            const tmp1 = results[i * 4 + 1];
            const tmp2 = results[i * 4 + 2];
            const tmp3 = results[i * 4 + 3];
            anchors[i] = [
                tmp0 - tmp2 / 2.0,
                tmp1 - tmp3 / 2.0,
                tmp0 + tmp2 / 2.0,
                tmp1 + tmp3 / 2.0
            ];
            if (anchors[i][0] < 0) {
                anchors[i][0] = 0;
            }
            if (anchors[i][0] > 1) {
                anchors[i][0] = 1;
            }
            if (anchors[i][1] < 0) {
                anchors[i][1] = 0;
            }
            if (anchors[i][1] > 1) {
                anchors[i][1] = 1;
            }
            if (anchors[i][2] < 0) {
                anchors[i][2] = 0;
            }
            if (anchors[i][2] > 1) {
                anchors[i][2] = 1;
            }
            if (anchors[i][3] < 0) {
                anchors[i][3] = 0;
            }
            if (anchors[i][3] > 1) {
                anchors[i][3] = 1;
            }
        }
        this.anchors = anchors;
    }

    getMaxIndex() {
        let maxIndex = -1;
        let maxConf = -1;
        let curConf = -2.0;
        const output_size = 10;

        for (let i = 0; i < this.anchor_num; i++) {
            curConf = sigm(this.modelResult[i * output_size + 1]);
            if (curConf > 0.55 && curConf > maxConf) {
                maxConf = curConf;
                maxIndex = i;
            }
        }
        this.maxIndex = maxIndex;
        function sigm(value) {
            return 1.0 / (1.0 + Math.exp(0.0 - value));
        }
    }

    decodeAnchor() {
        const index = this.maxIndex;
        const anchors = this.anchors;
        this.pxmin = anchors[index][0];
        this.pymin = anchors[index][1];
        this.pxmax = anchors[index][2];
        this.pymax = anchors[index][3];
    }

    decodeKp() {
        const modelResult = this.modelResult;
        const index = this.maxIndex;
        const px = (this.pxmin + this.pxmax) / 2;
        const py = (this.pymin + this.pymax) / 2;
        const pw = this.pxmax - this.pxmin;
        const ph = this.pymax - this.pymin;
        const prior_var = 0.1;
        const kp: number[][] = [[], [], []];
        kp[0][0] = (modelResult[index * this.output_size + 6] * pw * prior_var + px) * 256;
        kp[0][1] = (modelResult[index * this.output_size + 8] * ph * prior_var + py) * 256;
        kp[2][0] = (modelResult[index * this.output_size + 7] * pw * prior_var + px) * 256;
        kp[2][1] = (modelResult[index * this.output_size + 9] * ph * prior_var + py) * 256;
        this.kp = kp;
    }

    decodeBox() {
        const modelResult = this.modelResult;
        const output_size = this.output_size || 10;
        const pw = this.pxmax - this.pxmin;
        const ph = this.pymax - this.pymin;
        const px = this.pxmin + pw / 2;
        const py = this.pymin + ph / 2;
        const prior_var = 0.1;
        const index = this.maxIndex;

        const ox = modelResult[output_size * index + 2];
        const oy = modelResult[output_size * index + 3];
        const ow = modelResult[output_size * index + 4];
        const oh = modelResult[output_size * index + 5];

        const tx = ox * prior_var * pw + px;
        const ty = oy * prior_var * ph + py;
        const tw = this.tw = Math.pow(2.71828182, prior_var * ow) * pw;
        const th = this.th = Math.pow(2.71828182, prior_var * oh) * ph;
        const box: number[][] = [[], [], [], []];
        box[0][0] = (tx - tw / 2) * 256;
        box[0][1] = (ty - th / 2) * 256;
        box[1][0] = (tx + tw / 2) * 256;
        box[1][1] = (ty - th / 2) * 256;
        box[2][0] = (tx + tw / 2) * 256;
        box[2][1] = (ty + th / 2) * 256;
        box[3][0] = (tx - tw / 2) * 256;
        box[3][1] = (ty + th / 2) * 256;
        this.box = box;
    }

    decodeTriangle() {
        const box_enlarge = 1.04;
        const side = Math.max(this.tw * 256, this.th * 256) * (box_enlarge);
        const dir_v: number[] = [];
        const kp = this.kp;
        const triangle: number[][] = [[], [], []];
        const dir_v_r: number[] = [];

        dir_v[0] = kp[2][0] - kp[0][0];
        dir_v[1] = kp[2][1] - kp[0][1];
        const sq = Math.sqrt(Math.pow(dir_v[0], 2) + Math.pow(dir_v[1], 2)) || 1;
        dir_v[0] = dir_v[0] / sq;
        dir_v[1] = dir_v[1] / sq;

        dir_v_r[0] = dir_v[0] * 0 + dir_v[1] * 1;
        dir_v_r[1] = dir_v[0] * -1 + dir_v[1] * 0;
        triangle[0][0] = kp[2][0];
        triangle[0][1] = kp[2][1];
        triangle[1][0] = kp[2][0] + dir_v[0] * side;
        triangle[1][1] = kp[2][1] + dir_v[1] * side;
        triangle[2][0] = kp[2][0] + dir_v_r[0] * side;
        triangle[2][1] = kp[2][1] + dir_v_r[1] * side;
        this.triangle = triangle;
    }

    decodeSource() {
        const kp = this.kp;
        const box_shift = 0.0;
        const tmp0 = (kp[0][0] - kp[2][0]) * box_shift;
        const tmp1 = (kp[0][1] - kp[2][1]) * box_shift;
        const source: number[][] = [[], [], []];
        for (let i = 0; i < 3; i++) {
            source[i][0] = this.triangle[i][0] - tmp0;
            source[i][1] = this.triangle[i][1] - tmp1;
        }
        this.source = source;
    }

    outputResult() {
        for (let i = 0; i < 4; i++) {
            this.detectResult[i][0] = this.box[i][0];
            this.detectResult[i][1] = this.box[i][1];
        }
        this.detectResult[4][0] = this.kp[0][0];
        this.detectResult[4][1] = this.kp[0][1];
        this.detectResult[6][0] = this.kp[2][0];
        this.detectResult[6][1] = this.kp[2][1];

        for (let i = 0; i < 3; i++) {
            this.detectResult[i + 11][0] = this.source[i][0];
            this.detectResult[i + 11][1] = this.source[i][1];
        }
    }

    getaffinetransform() {
        // 图像上的原始坐标点。需要对所有坐标进行归一化，_x = (x - 128) / 128, _y = (128 - y) / 128
        // 坐标矩阵
        // x1 x2 x3
        // y1 y2 y3
        // z1 z2 z3
        const originPoints = [].concat(this.detectResult[11][0] as number / 128 - 1 as never)
            .concat(this.detectResult[12][0] / 128 - 1 as number as never)
            .concat(this.detectResult[13][0] / 128 - 1 as number as never)
            .concat(1 - this.detectResult[11][1] / 128 as number as never)
            .concat(1 - this.detectResult[12][1] / 128 as number as never)
            .concat(1 - this.detectResult[13][1] / 128 as number as never)
            .concat([1, 1, 1] as never[]);
        // originPoints = [0, 0, -1, .1, 1.1, 0.1, 1, 1, 1];
        const matrixA = new Matrix(3, 3, originPoints);
        // 转化后的点[128, 128, 0, 128, 0, 128] [0, 0, -1, 0, 1, 0]
        const matrixB = new Matrix(2, 3, [0, 0, -1, 0, -1, 0]);
        // M * A = B => M = B * A逆
        let _matrixA: any = inverseMatrix(matrixA.data);
        _matrixA = new Matrix(3, 3, _matrixA);
        this.mtr = Matrix_Product(matrixB, _matrixA);
    }

    async warpAffine() {
        const ctx = this.originCanvas.getContext('2d');
        this.originImageData = ctx!.getImageData(0, 0, 256, 256);
        const imageDataArr = await WarpAffine.main({
            imageData: this.originImageData,
            mtr: this.mtr.data,
            input: {
                width: 256,
                height: 256
            },
            output: {
                width: 224,
                height: 224
            }
        });
        this.imageData = new ImageData(Uint8ClampedArray.from(imageDataArr), 224, 224);
    }

    allReshapeToRGB(paddle) {
        const data = paddle.mediaProcessor.allReshapeToRGB(this.imageData, {
            gapFillWith: '#000',
            mean: [0, 0, 0],
            scale: 224,
            std: [1, 1, 1],
            targetShape: [1, 3, 224, 224],
            normalizeType: 1
        });
        this.feed = [{
            data: new Float32Array(data),
            shape: [1, 3, 224, 224],
            name: 'image',
            canvas: this.originImageData
        }];
    }
}

class Matrix {
    private row: number;
    private col: number;
    data: number[] | number[][];

    constructor(row, col, arr: number[] | number[][] = []) {
        // 行
        this.row = row;
        // 列
        this.col = col;
        if (arr[0] && arr[0] instanceof Array) {
            this.data = arr;
        }
        else {
            this.data = [];
            const _arr: number[] = [].concat(arr as never[]);
            // 创建row个元素的空数组
            const Matrix = new Array(row);
            // 对第一层数组遍历
            for (let i = 0; i < row; i++) {
                // 每一行创建col列的空数组
                Matrix[i] = new Array(col).fill('');
                Matrix[i].forEach((_item, index, cur) => {
                    cur[index] = _arr.shift() || 0;
                });
            }
            // 将矩阵保存到this.data上
            this.data = Matrix;
        }
    }
}

const Matrix_Product = (A, B) => {
    const tempMatrix = new Matrix(A.row, B.col);
    if (A.col === B.row) {
        for (let i = 0; i < A.row; i++) {
            for (let j = 0; j < B.col; j++) {
                tempMatrix.data[i][j] = 0;
                for (let n = 0; n < A.col; n++) {
                    tempMatrix.data[i][j] += A.data[i][n] * B.data[n][j];
                }
                tempMatrix.data[i][j] = tempMatrix.data[i][j].toFixed(5);
            }
        }
        return tempMatrix;
    }
    return false;
};

// 求行列式
const determinant = matrix => {
    const order = matrix.length;
    let cofactor;
    let result = 0;
    if (order === 1) {
        return matrix[0][0];
    }
    for (let i = 0; i < order; i++) {
        cofactor = [];
        for (let j = 0; j < order - 1; j++) {
            cofactor[j] = [];
            for (let k = 0; k < order - 1; k++) {
                cofactor[j][k] = matrix[j + 1][k < i ? k : k + 1];
            }
        }
        result += matrix[0][i] * Math.pow(-1, i) * determinant(cofactor);
    }
    return result;
};

// 矩阵数乘
function scalarMultiply(num, matrix) {
    const row = matrix.length;
    const col = matrix[0].length;
    const result: number[][] = [];
    for (let i = 0; i < row; i++) {
        result[i] = [];
        for (let j = 0; j < col; j++) {
            result[i][j] = num * matrix[i][j];
        }
    }
    return result;
}

// 求逆矩阵
function inverseMatrix(matrix) {
    if (determinant(matrix) === 0) {
        return false;
    }
    // 求代数余子式
    function cofactor(matrix, row, col) {
        const order = matrix.length;
        const new_matrix: number[][] = [];
        let _row;
        let _col;
        for (let i = 0; i < order - 1; i++) {
            new_matrix[i] = [];
            _row = i < row ? i : i + 1;
            for (let j = 0; j < order - 1; j++) {
                _col = j < col ? j : j + 1;
                new_matrix[i][j] = matrix[_row][_col];
            }
        }
        return Math.pow(-1, row + col) * determinant(new_matrix);
    }
    const order = matrix.length;
    const adjoint: number[][] = [];
    for (let i = 0; i < order; i++) {
        adjoint[i] = [];
        for (let j = 0; j < order; j++) {
            adjoint[i][j] = cofactor(matrix, j, i);
        }
    }
    return scalarMultiply(1 / determinant(matrix), adjoint);
}

export default DetectProcess;
