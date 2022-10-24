import clipper from 'js-clipper';
import { divide, enableBoundaryChecking, plus } from 'number-precision';
import CV from '@paddlejs-mediapipe/opencv/library/opencv_ocr';
import {clip} from './util';

import * as d3Polygon from 'd3-polygon';
import { BOX, POINT, POINTS } from "./type";

export default class DBPostprocess {
    private readonly thresh: number;
    private readonly box_thresh: number;
    private readonly max_candidates: number;
    private readonly unclip_ratio: number;
    private readonly min_size: number;
    private readonly pred: number[];
    private readonly segmentation: number[];
    private readonly width: number;
    private readonly height: number;

    constructor(result: number[], shape: number[], thresh = 0.3, box_thresh = 0.6, unclip_ratio = 1.5) {
        enableBoundaryChecking(false);
        this.thresh = thresh ? thresh : 0.3;
        this.box_thresh = box_thresh ? box_thresh : 0.6;
        this.max_candidates = 1000;
        this.unclip_ratio = unclip_ratio ? unclip_ratio:1.5;
        this.min_size = 3;
        this.width = shape[0];
        this.height = shape[1];
        this.pred = result;
        this.segmentation = [];
        this.pred.forEach((item: number) => {
            this.segmentation.push(item > this.thresh ? 255 : 0);
        });
    }

    public outputBox() {
        // eslint-disable-next-line new-cap
        const src = new CV.matFromArray(this.width, this.height, CV.CV_8UC1, this.segmentation);
        const contours = new CV.MatVector();
        const hierarchy = new CV.Mat();
        // 获取轮廓
        CV.findContours(src, contours, hierarchy, CV.RETR_LIST, CV.CHAIN_APPROX_SIMPLE);
        const num_contours = Math.min(contours.size(), this.max_candidates);
        const boxes: BOX = [];
        const scores: number[] = [];
        const arr: number[] = [];
        for (let i = 0; i < num_contours; i++) {
            const contour = contours.get(i);
            const minBox = this.get_mini_boxes(contour);
            const points = minBox.points;
            let side = minBox.side;
            if (side < this.min_size) {
                continue;
            }
            const score = this.box_score_fast(this.pred, points);
            if (this.box_thresh > score) {
                continue;
            }
            let box = this.unclip(points);
            // eslint-disable-next-line new-cap
            const boxMap = new CV.matFromArray(box.length / 2, 1, CV.CV_32SC2, box);
            const resultObj = this.get_mini_boxes(boxMap);
            box = resultObj.points as [number, number][];
            side = resultObj.side;
            if (side < this.min_size + 2) {
                continue;
            }
            box.forEach(item => {
                item[0] = clip(Math.round(item[0]), 0, this.width);
                item[1] = clip(Math.round(item[1]), 0, this.height);
            });
            boxes.push(box);
            scores.push(score);
            arr.push(i);
            boxMap.delete();
        }
        src.delete();
        contours.delete();
        hierarchy.delete();
        return { boxes, scores };
    }

    private get_mini_boxes(contour: any) {
        // 生成最小外接矩形
        const bounding_box = CV.minAreaRect(contour);
        const points: POINTS = [];
        const mat = new CV.Mat();
        // 获取矩形的四个顶点坐标
        CV.boxPoints(bounding_box, mat);
        for (let i = 0; i < mat.data32F.length; i += 2) {
            const arr: POINT = [mat.data32F[i], mat.data32F[i + 1]];
            points.push(arr);
        }
        function sortNumber(a: POINT, b: POINT) {
            return a[0] - b[0];
        }
        points.sort(sortNumber);

        let index_1: number;
        let index_2: number;
        let index_3: number;
        let index_4: number;
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

        return { points: box, side };
    }

    private box_score_fast(bitmap: number[], _box: POINTS) {
        const h = this.height;
        const w = this.width;
        const box = JSON.parse(JSON.stringify(_box));
        const x = [] as number[];
        const y = [] as number[];
        box.forEach((item: POINT) => {
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
        box.forEach((item: POINT) => {
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
            sliceArr.push(...bitmap.slice(this.width * i + xmin, this.height * i + xmax + 1) as []);
        }
        const mean = this.mean(sliceArr, mask.data);
        mask.delete();
        points.delete();
        pts.delete();
        return mean;
    }

    private unclip(box: POINTS) {
        const unclip_ratio = this.unclip_ratio;
        const area = Math.abs(d3Polygon.polygonArea(box as [number, number][]));
        const length = d3Polygon.polygonLength(box as [number, number][]);
        const distance = area * unclip_ratio / length;
        const tmpArr: { X: number; Y: number; }[] = [];
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
        const expanded: { X: number; Y: number; }[][] = [];
        offset.Execute(expanded, distance);
        let expandedArr: POINTS = [];
        expanded[0] && expanded[0].forEach(item => {
            expandedArr.push([item.X, item.Y]);
        });
        expandedArr = [].concat(...expandedArr as []);
        return expandedArr;
    }

    private mean(data: number[], mask: number[]) {
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
}
