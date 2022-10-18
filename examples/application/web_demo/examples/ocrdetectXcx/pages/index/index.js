/* global wx, Page */
import * as paddlejs from '@paddlejs/paddlejs-core';
import '@paddlejs/paddlejs-backend-webgl';
import clipper from 'js-clipper';
import { divide, enableBoundaryChecking, plus } from 'number-precision';
// eslint-disable-next-line no-undef
const plugin = requirePlugin('paddlejs-plugin');
const Polygon = require('d3-polygon');

global.wasm_url = 'pages/index/wasm/opencv_js.wasm.br';
const CV = require('./wasm/opencv.js');

plugin.register(paddlejs, wx);

const imgList = [
    'https://paddlejs.bj.bcebos.com/xcx/ocr.png',
    './img/width.png'
];

// eslint-disable-next-line max-lines-per-function
const outputBox = res => {
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

let detectRunner;

Page({
    data: {
        imgList: imgList,
        imgInfo: {},
        result: '',
        loaded: false
    },

    onLoad() {
        enableBoundaryChecking(false);
        const me = this;
        detectRunner = new paddlejs.Runner({
            modelPath: 'https://paddleocr.bj.bcebos.com/PaddleJS/PP-OCRv3/ch/ch_PP-OCRv3_det_infer_js_640/model.json',
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            bgr: true,
            webglFeedProcess: true
        });
        detectRunner.init().then(_ => {
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
            success: imgInfo => {
                const {
                    path,
                    width,
                    height
                } = imgInfo;

                const canvasPath = imgPath.includes('http') ? path : imgPath;
                const canvasId = 'myCanvas';
                const ctx = wx.createCanvasContext(canvasId);
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
                ctx.drawImage(canvasPath, x, y, sw, sh);
                ctx.draw(false, () => {
                    // API 1.9.0 获取图像数据
                    wx.canvasGetImageData({
                        canvasId: canvasId,
                        x: 0,
                        y: 0,
                        width: 960,
                        height: 960,
                        success(res) {
                            me.predict({
                                data: res.data,
                                width: 960,
                                height: 960
                            }, {
                                canvasPath,
                                sw,
                                sh,
                                x,
                                y
                            });
                        }
                    });
                });
            }
        });
    },

    predict(res, img) {
        const me = this;
        detectRunner.predict(res, function (data) {
            // 获取坐标
            const points = outputBox(data);
            me.drawCanvasPoints(img, points.boxes);
            me.setData({
                result: JSON.stringify(points.boxes)
            });
        });
    },

    drawCanvasPoints(img, points) {
        const canvasId = 'result';
        const ctx = wx.createCanvasContext(canvasId);
        ctx.drawImage(img.canvasPath, img.x, img.y, img.sw, img.sh);
        points.length && points.forEach(point => {
            // 开始一个新的绘制路径
            ctx.beginPath();
            // 设置线条颜色为蓝色
            ctx.strokeStyle = 'blue';
            // 设置路径起点坐标
            ctx.moveTo(point[0][0], point[0][1]);
            ctx.lineTo(point[1][0], point[1][1]);
            ctx.lineTo(point[2][0], point[2][1]);
            ctx.lineTo(point[3][0], point[3][1]);
            ctx.closePath();
            ctx.stroke();
        });
        ctx.draw();
    }
});
