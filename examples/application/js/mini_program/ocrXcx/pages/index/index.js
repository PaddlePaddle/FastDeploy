/* global wx, Page */
import * as paddlejs from '@paddlejs/paddlejs-core';
import '@paddlejs/paddlejs-backend-webgl';
import clipper from 'js-clipper';
import { divide, enableBoundaryChecking, plus } from 'number-precision';

import { recDecode } from 'recPostprocess.js';
// eslint-disable-next-line no-undef
const plugin = requirePlugin('paddlejs-plugin');
const Polygon = require('d3-polygon');

global.wasm_url = 'pages/index/wasm/opencv_js.wasm.br';
const CV = require('./wasm/opencv.js');

plugin.register(paddlejs, wx);

let DETSHAPE = 960;
let RECWIDTH = 320;
const RECHEIGHT = 32;

// 声明后续图像变换要用到的canvas；此时未绑定
let canvas_det;
let canvas_rec;
let my_canvas;
let my_canvas_ctx;


const imgList = [
    'https://paddlejs.bj.bcebos.com/xcx/ocr.png'
];

// eslint-disable-next-line max-lines-per-function
const outputBox = (res) => {
    const thresh = 0.3;
    const box_thresh = 0.5;
    const max_candidates = 1000;
    const min_size = 3;
    const width = 960;
    const height = 960;
    const pred = res;
    const segmentation = [];
    pred.forEach(item => {
        segmentation.push(item > thresh ? 255 : 0);
    });

    function get_mini_boxes(contour) {
        // 生成最小外接矩形
        const bounding_box = CV.minAreaRect(contour);
        const points = [];
        const mat = new CV.Mat();
        // 获取矩形的四个顶点坐标
        CV.boxPoints(bounding_box, mat);
        for (let i = 0; i < mat.data32F.length; i += 2) {
            const arr = [];
            arr[0] = mat.data32F[i];
            arr[1] = mat.data32F[i + 1];
            points.push(arr);
        }

        function sortNumber(a, b) {
            return a[0] - b[0];
        }
        points.sort(sortNumber);
        let index_1 = 0;
        let index_2 = 1;
        let index_3 = 2;
        let index_4 = 3;
        if (points[1][1] > points[0][1]) {
            index_1 = 0;
            index_4 = 1;
        }
        else {
            index_1 = 1;
            index_4 = 0;
        }

        if (points[3][1] > points[2][1]) {
            index_2 = 2;
            index_3 = 3;
        }
        else {
            index_2 = 3;
            index_3 = 2;
        }
        const box = [
            points[index_1],
            points[index_2],
            points[index_3],
            points[index_4]
        ];
        const side = Math.min(bounding_box.size.height, bounding_box.size.width);
        mat.delete();
        return {
            points: box,
            side
        };
    }

    function box_score_fast(bitmap, _box) {
        const h = height;
        const w = width;
        const box = JSON.parse(JSON.stringify(_box));
        const x = [];
        const y = [];
        box.forEach(item => {
            x.push(item[0]);
            y.push(item[1]);
        });
        // clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
        const xmin = clip(Math.floor(Math.min(...x)), 0, w - 1);
        const xmax = clip(Math.ceil(Math.max(...x)), 0, w - 1);
        const ymin = clip(Math.floor(Math.min(...y)), 0, h - 1);
        const ymax = clip(Math.ceil(Math.max(...y)), 0, h - 1);
        // eslint-disable-next-line new-cap
        const mask = new CV.Mat.zeros(ymax - ymin + 1, xmax - xmin + 1, CV.CV_8UC1);
        box.forEach(item => {
            item[0] = Math.max(item[0] - xmin, 0);
            item[1] = Math.max(item[1] - ymin, 0);
        });
        const npts = 4;
        const point_data = new Uint8Array(box.flat());
        const points = CV.matFromArray(npts, 1, CV.CV_32SC2, point_data);
        const pts = new CV.MatVector();
        pts.push_back(points);
        const color = new CV.Scalar(255);
        // 多个多边形填充
        CV.fillPoly(mask, pts, color, 1);
        const sliceArr = [];
        for (let i = ymin; i < ymax + 1; i++) {
            sliceArr.push(...bitmap.slice(960 * i + xmin, 960 * i + xmax + 1));
        }
        const mean = num_mean(sliceArr, mask.data);
        mask.delete();
        points.delete();
        pts.delete();
        return mean;
    }

    function clip(data, min, max) {
        return data < min ? min : data > max ? max : data;
    }

    function unclip(box) {
        const unclip_ratio = 1.6;
        const area = Math.abs(Polygon.polygonArea(box));
        const length = Polygon.polygonLength(box);
        const distance = area * unclip_ratio / length;
        const tmpArr = [];
        box.forEach(item => {
            const obj = {
                X: 0,
                Y: 0
            };
            obj.X = item[0];
            obj.Y = item[1];
            tmpArr.push(obj);
        });
        const offset = new clipper.ClipperOffset();
        offset.AddPath(tmpArr, clipper.JoinType.jtRound, clipper.EndType.etClosedPolygon);
        const expanded = [];
        offset.Execute(expanded, distance);
        let expandedArr = [];
        expanded[0] && expanded[0].forEach(item => {
            expandedArr.push([item.X, item.Y]);
        });
        expandedArr = [].concat(...expandedArr);
        return expandedArr;
    }

    function num_mean(data, mask) {
        let sum = 0;
        let length = 0;
        for (let i = 0; i < data.length; i++) {
            if (mask[i]) {
                sum = plus(sum, data[i]);
                length++;
            }
        }
        return divide(sum, length);
    }

    // eslint-disable-next-line new-cap
    const src = new CV.matFromArray(960, 960, CV.CV_8UC1, segmentation);
    const contours = new CV.MatVector();
    const hierarchy = new CV.Mat();
    // 获取轮廓
    CV.findContours(src, contours, hierarchy, CV.RETR_LIST, CV.CHAIN_APPROX_SIMPLE);
    const num_contours = Math.min(contours.size(), max_candidates);
    const boxes = [];
    const scores = [];
    const arr = [];
    for (let i = 0; i < num_contours; i++) {
        const contour = contours.get(i);
        let {
            points,
            side
        } = get_mini_boxes(contour);
        if (side < min_size) {
            continue;
        }
        const score = box_score_fast(pred, points);
        if (box_thresh > score) {
            continue;
        }
        let box = unclip(points);
        // eslint-disable-next-line new-cap
        const boxMap = new CV.matFromArray(box.length / 2, 1, CV.CV_32SC2, box);
        const resultObj = get_mini_boxes(boxMap);
        box = resultObj.points;
        side = resultObj.side;
        if (side < min_size + 2) {
            continue;
        }
        box.forEach(item => {
            item[0] = clip(Math.round(item[0]), 0, 960);
            item[1] = clip(Math.round(item[1]), 0, 960);
        });
        boxes.push(box);
        scores.push(score);
        arr.push(i);
        boxMap.delete();
    }
    src.delete();
    contours.delete();
    hierarchy.delete();
    return {
        boxes,
        scores
    };
};

const sorted_boxes = (box) => {
  function sortNumber(a, b) {
      return a[0][1] - b[0][1];
  }

  const boxes = box.sort(sortNumber);
  const num_boxes = boxes.length;
  for (let i = 0; i < num_boxes - 1; i++) {
      if (Math.abs(boxes[i + 1][0][1] - boxes[i][0][1]) < 10
          && boxes[i + 1][0][0] < boxes[i][0][0]) {
          const tmp = boxes[i];
          boxes[i] = boxes[i + 1];
          boxes[i + 1] = tmp;
      }
  }
  return boxes;
}

function flatten(arr) {
  return arr.toString().split(',').map(item => +item);
}

function int(num) {
  return num > 0 ? Math.floor(num) : Math.ceil(num);
}

function clip(data, min, max) {
  return data < min ? min : data > max ? max : data;
}

function get_rotate_crop_image(img, points) {
  const img_crop_width = int(Math.max(
      linalg_norm(points[0], points[1]),
      linalg_norm(points[2], points[3])
  ));
  const img_crop_height = int(Math.max(
      linalg_norm(points[0], points[3]),
      linalg_norm(points[1], points[2])
  ));
  const pts_std = [
      [0, 0],
      [img_crop_width, 0],
      [img_crop_width, img_crop_height],
      [0, img_crop_height]
  ];
  const srcTri = CV.matFromArray(4, 1, CV.CV_32FC2, flatten(points));
  const dstTri = CV.matFromArray(4, 1, CV.CV_32FC2, flatten(pts_std));
  // 获取到目标矩阵
  const M = CV.getPerspectiveTransform(srcTri, dstTri);
  const src = CV.imread(img);
  const dst = new CV.Mat();
  const dsize = new CV.Size(img_crop_width, img_crop_height);
  // 透视转换
  CV.warpPerspective(src, dst, M, dsize, CV.INTER_CUBIC, CV.BORDER_REPLICATE, new CV.Scalar());

  const dst_img_height = dst.rows;
  const dst_img_width = dst.cols;
  let dst_rot;
  // 图像旋转
  if (dst_img_height / dst_img_width >= 1.5) {
      dst_rot = new CV.Mat();
      const dsize_rot = new CV.Size(dst.rows, dst.cols);
      const center = new CV.Point(dst.cols / 2, dst.cols / 2);
      const M = CV.getRotationMatrix2D(center, 90, 1);
      CV.warpAffine(dst, dst_rot, M, dsize_rot, CV.INTER_CUBIC, CV.BORDER_REPLICATE, new CV.Scalar());
  }

  const dst_resize = new CV.Mat();
  const dsize_resize = new CV.Size(0, 0);
  let scale;
  if (dst_rot) {
      scale = RECHEIGHT / dst_rot.rows;
      CV.resize(dst_rot, dst_resize, dsize_resize, scale, scale, CV.INTER_AREA);
      dst_rot.delete();
  }
  else {
      scale = RECHEIGHT / dst_img_height;
      CV.resize(dst, dst_resize, dsize_resize, scale, scale, CV.INTER_AREA);
  }

  canvas_det.width = dst_resize.cols;
  canvas_det.height = dst_resize.rows;
  canvas_det.getContext('2d').clearRect(0, 0, canvas_det.width, canvas_det.height);
  CV.imshow(canvas_det, dst_resize);

  src.delete();
  dst.delete();
  dst_resize.delete();
  srcTri.delete();
  dstTri.delete();
}

function linalg_norm(x, y) {
  return Math.sqrt(Math.pow(x[0] - y[0], 2) + Math.pow(x[1] - y[1], 2));
}

function resize_norm_img_splice(
  image,
  origin_width,
  origin_height,
  index = 0
) {
  canvas_rec.width = RECWIDTH;
  canvas_rec.height = RECHEIGHT;
  const ctx = canvas_rec.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.clearRect(0, 0, canvas_rec.width, canvas_rec.height);
  // ctx.drawImage(image, -index * RECWIDTH, 0, origin_width, origin_height);
  ctx.putImageData(image, -index * RECWIDTH, 0);
}

// 声明检测和识别Runner；未初始化
let detectRunner;
let recRunner;

Page({
    data: {
        photo_src:'',
        imgList: imgList,
        imgInfo: {},
        result: '',
        select_mode: false,
        loaded: false
    },
    switch_choose(){
      this.setData({
        select_mode: true
      })
    },
    switch_example(){
      this.setData({
        select_mode: false
      })
    },
    chose_photo:function(evt){
      let _this = this
      wx.chooseImage({
        count: 1,
        sizeType: ['original', 'compressed'],
        sourceType: ['album', 'camera'],
        success(res) {
          console.log(res.tempFilePaths)               //一个数组，每个元素都是“http://...”图片地址
          _this.setData({
            photo_src: res.tempFilePaths[0]
          })
        }
      })
    },
    reselect:function(evt){
      let _this = this
      wx.chooseImage({
        count: 1,
        sizeType: ['original', 'compressed'],
        sourceType: ['album', 'camera'],
        success(res) {
          _this.setData({
            photo_src: res.tempFilePaths[0]
          })
        }
      })
    },
    photo_preview:function(evt){
      let _this = this;
      let imgs = [];
      imgs.push(_this.data.photo_src);
      wx.previewImage({
        urls:imgs
      })
    },

    predect_choose_img() {
      console.log(this.data.photo_src)
      this.getImageInfo(this.data.photo_src);
    },

    onLoad() {
      enableBoundaryChecking(false);
      // 绑定canvas；该操作是异步，因此最好加延迟保证后续使用时已完成绑定
      wx.createSelectorQuery()
        .select('#canvas_det')
        .fields({ node: true, size: true })
        .exec(async(res) => {
          canvas_det = res[0].node;
        });

      wx.createSelectorQuery()
      .select('#canvas_rec')
      .fields({ node: true, size: true })
      .exec(async(res) => {
        canvas_rec = res[0].node;
      });

      wx.createSelectorQuery()
        .select('#myCanvas')
        .fields({ node: true, size: true })
        .exec((res) => {
            my_canvas = res[0].node;
            my_canvas_ctx = my_canvas.getContext('2d');
        });
        
      const me = this;
      // 初始化Runner
      detectRunner = new paddlejs.Runner({
          modelPath: 'https://paddleocr.bj.bcebos.com/PaddleJS/PP-OCRv3/ch/ch_PP-OCRv3_det_infer_js_960/model.json',
          mean: [0.485, 0.456, 0.406],
          std: [0.229, 0.224, 0.225],
          bgr: true,
          webglFeedProcess: true
      });
      recRunner = new paddlejs.Runner({
          modelPath: 'https://paddleocr.bj.bcebos.com/PaddleJS/PP-OCRv3/ch/ch_PP-OCRv3_rec_infer_js/model.json',
          fill: '#000',
          mean: [0.5, 0.5, 0.5],
          std: [0.5, 0.5, 0.5],
          bgr: true,
          webglFeedProcess: true
      });
      // 等待模型数据全部加载完成
      Promise.all([detectRunner.init(), recRunner.init()]).then(_ => {
            me.setData({
                loaded: true
            });
      });
        
    },

    selectImage(event) {
        const imgPath = this.data.imgList[event.target.dataset.index];
        this.getImageInfo(imgPath);
    },

    getImageInfo(imgPath) {
        const me = this;
        wx.getImageInfo({
            src: imgPath,
            success: (imgInfo) => {
                const {
                    path,
                    width,
                    height
                } = imgInfo;
                const canvasPath = imgPath.includes('http') ? path : imgPath;

                let sw = 960;
                let sh = 960;
                let x = 0;
                let y = 0;

                if (height / width >= 1) {
                    sw = Math.round(sh * width / height);
                    x = Math.floor((960 - sw) / 2);
                }
                else {
                    sh = Math.round(sw * height / width);
                    y = Math.floor((960 - sh) / 2);
                }
                my_canvas.width = sw;
                my_canvas.height = sh;

                // 微信上canvas输入图片
                const image = my_canvas.createImage();
                image.src = canvasPath;
                image.onload = () => {
                    my_canvas_ctx.clearRect(0, 0, my_canvas.width, my_canvas.height);
                    my_canvas_ctx.drawImage(image, x, y, sw, sh);
                    const imageData = my_canvas_ctx.getImageData(0, 0, sw, sh);
                    // 开始识别
                    me.recognize({
                        data: imageData.data,
                        width: 960,
                        height: 960
                    }, {canvasPath, sw, sh, x, y});
                }
            }
        });
    },

    async recognize(res, img) {
      const me = this;
      // 文本框选坐标点
      let points;
      await detectRunner.predict(res, function (detectRes) {
        points = outputBox(detectRes);
      });

      // 绘制文本框
      me.drawCanvasPoints(img, points.boxes);

      // 排序，使得最后结果输出尽量按照从上到下的顺序
      const boxes = sorted_boxes(points.boxes);

      const text_list = [];

      for (let i = 0; i < boxes.length; i++) {
          const tmp_box = JSON.parse(JSON.stringify(boxes[i]));
          // 获取tmp_box对应图片到canvas_det
          get_rotate_crop_image(res, tmp_box);
          // 这里是计算要识别文字的图片片段是否大于识别模型要求的输入宽度；超过了的话会分成多次识别，再拼接结果
          const width_num = Math.ceil(canvas_det.width / RECWIDTH);

          let text_list_tmp = '';
          for (let j = 0; j < width_num; j++) {
            // 根据原图的宽度进行裁剪拼接，超出指定宽度会被截断；然后再次识别，最后拼接起来
            resize_norm_img_splice(canvas_det.getContext('2d').getImageData(0, 0, canvas_det.width, canvas_det.height), canvas_det.width, canvas_det.height, j);

            const imgData = canvas_rec.getContext('2d').getImageData(0, 0, canvas_rec.width, canvas_rec.height);

            await recRunner.predict(imgData, function(output){
              // 将输出向量转化为idx再传化为对应字符
              const text = recDecode(output);
              text_list_tmp = text_list_tmp.concat(text.text);
            });
          }
          text_list.push(text_list_tmp);
      }
      me.setData({
        result: JSON.stringify(boxes) + JSON.stringify(text_list)
      });
    },

    drawCanvasPoints(img, points) {
        // 设置线条
        my_canvas_ctx.strokeStyle = 'blue';
        my_canvas_ctx.lineWidth = 5;

        // 先绘制图片
        const image = my_canvas.createImage();
        image.src = img.canvasPath;
        image.onload = () => {
            my_canvas_ctx.clearRect(0, 0, my_canvas_ctx.width, my_canvas_ctx.height);
            my_canvas_ctx.drawImage(image, img.x, img.y, img.sw, img.sh);
            // 绘制线框
            points.length && points.forEach(point => {
                my_canvas_ctx.beginPath();
                // 设置路径起点坐标
                my_canvas_ctx.moveTo(point[0][0], point[0][1]);
                my_canvas_ctx.lineTo(point[1][0], point[1][1]);
                my_canvas_ctx.lineTo(point[2][0], point[2][1]);
                my_canvas_ctx.lineTo(point[3][0], point[3][1]);
                my_canvas_ctx.lineTo(point[0][0], point[0][1]);
                my_canvas_ctx.stroke();
                my_canvas_ctx.closePath();
            });
        }

    }
});
