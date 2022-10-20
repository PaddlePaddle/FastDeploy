import { character } from 'ppocr_keys_v1.js';

const ocr_character = character;
let preds_idx = [];
let preds_prob = [];

function init(preds) {
  preds_idx = [];
  preds_prob = [];
  // preds: [1, ?, 6625]
  const pred_len = 6625;
  for (let i = 0; i < preds.length; i += pred_len) {
      const tmpArr = preds.slice(i, i + pred_len - 1);
      const tmpMax = Math.max(...tmpArr);
      const tmpIdx = tmpArr.indexOf(tmpMax);
      preds_prob.push(tmpMax);
      preds_idx.push(tmpIdx);
  }
}

function get_ignored_tokens() {
  return [0];
}

function decode(text_index, text_prob, is_remove_duplicate = false) {
  const ignored_tokens = get_ignored_tokens();
  const char_list = [];
  const conf_list = [];
  for (let idx = 0; idx < text_index.length; idx++) {
      if (text_index[idx] in ignored_tokens) {
          continue;
      }
      if (is_remove_duplicate) {
          if (idx > 0 && text_index[idx - 1] === text_index[idx]) {
              continue;
          }
      }
      char_list.push(ocr_character[text_index[idx] - 1]);
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

export function recDecode(preds) {
  init(preds);
  return decode(preds_idx, preds_prob, true);
}