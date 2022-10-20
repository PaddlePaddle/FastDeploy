import character from './ppocr_keys_v1.txt';

export default class recPostprocess {
    private ocr_character: string[];
    private preds_idx: number[];
    private preds_prob: number[];
    // preds: [1, ?, 6625]
    constructor(preds: number[]) {
        this.ocr_character = character.toString().split('\n');
        const preds_idx: number[] = [];
        const preds_prob: number[] = [];
        const pred_len = 6625;
        for (let i = 0; i < preds.length; i += pred_len) {
            const tmpArr = preds.slice(i, i + pred_len - 1);
            const tmpMax = Math.max(...tmpArr);
            const tmpIdx = tmpArr.indexOf(tmpMax);
            preds_prob.push(tmpMax);
            preds_idx.push(tmpIdx);
        }
        this.preds_idx = preds_idx;
        this.preds_prob = preds_prob;
    }

    private decode(text_index: number[], text_prob: number[], is_remove_duplicate = false) {
        const ignored_tokens = this.get_ignored_tokens();
        const char_list: string[] = [];
        const conf_list: number[] = [];
        for (let idx = 0; idx < text_index.length; idx++) {
            if (text_index[idx] in ignored_tokens) {
                continue;
            }
            if (is_remove_duplicate) {
                if (idx > 0 && text_index[idx - 1] === text_index[idx]) {
                    continue;
                }
            }
            char_list.push(this.ocr_character[text_index[idx] - 1]);
            if (text_prob) {
                conf_list.push(text_prob[idx]);
            }
            else {
                conf_list.push(1);
            }
        }
        let text = '';
        let mean = 0;
        if (char_list.length) {
            text = char_list.join('');
            let sum = 0;
            conf_list.forEach(item => {
                sum += item;
            });
            mean = sum / conf_list.length;
        }
        return { text, mean };
    }

    private get_ignored_tokens() {
        return [0];
    }

    public outputResult() {
        return this.decode(this.preds_idx, this.preds_prob, true);
    }
}
