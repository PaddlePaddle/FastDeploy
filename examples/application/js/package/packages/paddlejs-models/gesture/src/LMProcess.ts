export default class LMProcess {
    private input_n: number;
    private input_h: number;
    private input_w: number;
    private input_c: number;
    private class_dim: number;
    private points_num: number;
    private result: any;
    private conf!: number;
    private kp: any;
    private forefinger: any;
    private output_softmax: any;
    type!: string;

    constructor(result) {
        /*
        * result[0] :是否为手
        * result[1:43]: 手的21个关键点
        * result[43:49]: 几中手势分类，包含但不限于石头剪刀布，为了提升准确度
        * this.kp decode result得出的21个手指关键点，this.kp[8]为食指
        * this.conf 是否为手，越大，越可能是手
        */
        this.input_n = 1;
        this.input_h = 224;
        this.input_w = 224;
        this.input_c = 3;
        this.class_dim = 6;
        this.points_num = 1;
        this.result = result;
    }

    sigm(value) {
        return 1.0 / (1.0 + Math.exp(0.0 - value));
    }

    decodeConf() {
        this.conf = this.sigm(this.result[0]);
    }

    decodeKp() {
        // 21个关键点，下标1开始
        const offset = 1;
        const result = this.result;
        this.kp = [];
        for (let i = 0; i < this.points_num; i++) {
            const arr: number[] = [];
            arr.push((result[offset + i * 2] + 0.5) * this.input_h);
            arr.push((result[offset + i * 2 + 1] + 0.5) * this.input_h);
            this.kp.push(arr);
        }
        this.forefinger = this.kp[0];
    }

    softMax() {
        let max = 0;
        let sum = 0;
        const offset = 2;
        const class_dim = this.class_dim = 7;
        const result = this.result;
        const output_softmax = new Array(7).fill(null);

        for (let i = 0; i < class_dim; i++) {
            if (max < result[i + offset]) {
                max = result[i + offset];
            }
        }

        for (let i = 0; i < class_dim; i++) {
            output_softmax[i] = Math.exp(result[i + offset] - max);
            sum += output_softmax[i];
        }

        for (let i = 0; i < class_dim; i++) {
            output_softmax[i] /= sum;
        }

        this.output_softmax = output_softmax;
    }

    output() {
        this.decodeKp();
        this.softMax();

        let label_index = 0;
        let max_pro = this.output_softmax[0];
        for (let i = 1; i < this.class_dim; i++) {
            if (max_pro < this.output_softmax[i]) {
                label_index = i;
                max_pro = this.output_softmax[i];
            }
        }
        // 最后一位：有无手
        if (label_index !== 0 && label_index !== this.class_dim - 1 && max_pro > 0.9) {
            const ges = ['其他', '布', '剪刀', '石头', '1', 'ok'];
            this.type = ges[label_index];
            return;
        }
        this.type = 'other';
    }
}
