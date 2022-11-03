package com.baidu.paddle.fastdeploy.app.ui.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Point;
import android.graphics.Rect;
import android.os.Handler;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.view.View;

import com.baidu.paddle.fastdeploy.app.ui.util.StringUtil;
import com.baidu.paddle.fastdeploy.app.ui.view.model.BasePolygonResultModel;
import com.baidu.paddle.fastdeploy.app.ui.view.model.PoseViewResultModel;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Created by ruanshimin on 2018/5/15.
 */

public class ResultMaskView extends View {
    private static final int[][] COLOR_MAP_DEF = {
            {0, 0, 0},
            {244, 35, 232},
            {70, 70, 70},
            {102, 102, 156},
            {190, 153, 153},
            {153, 153, 153},
            {250, 170, 30},
            {220, 220, 0},
            {107, 142, 35},
            {152, 251, 152},
            {70, 130, 180},
            {220, 20, 60},
            {255, 0, 0},
            {0, 0, 142},
            {0, 0, 70},
            {0, 60, 100},
            {0, 80, 100},
            {0, 0, 230},
            {119, 11, 32},
            {128, 64, 128}
    };

    private float sizeRatio;
    private List<BasePolygonResultModel> mResultModelList;
    private Point originPt = new Point();
    private int imgWidth;
    private int imgHeight;
    private Paint textPaint;

    private Handler handler;

    public void setHandler(Handler mHandler) {
        handler = mHandler;
    }

    public ResultMaskView(Context context) {
        super(context);
    }

    private Map<Integer, Paint> paintFixPool = new HashMap<>();

    public ResultMaskView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);

    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        setLayerType(View.LAYER_TYPE_SOFTWARE, null);
    }


    public void setPolygonListInfo(List<BasePolygonResultModel> modelList, int width, int height) {
        imgWidth = width;
        imgHeight = height;
        mResultModelList = modelList;
        handler.post(new Runnable() {
            @Override
            public void run() {
                invalidate();
            }
        });
    }


    public void clear() {
        mResultModelList = null;
        handler.post(new Runnable() {
            @Override
            public void run() {
                invalidate();
            }
        });
    }

    private void preCaculate() {
        float ratio = (float) getMeasuredWidth() / (float) getMeasuredHeight();
        float ratioBitmap = (float) imgWidth / (float) imgHeight;
        // | |#####| |模式
        if (ratioBitmap < ratio) {
            sizeRatio = (float) getMeasuredHeight() / (float) imgHeight;
            int x = (int) (getMeasuredWidth() - sizeRatio * imgWidth) / 2;
            originPt.set(x, 0);
        } else {
            // ------------
            //
            // ------------
            // ############
            // ------------
            //
            // ------------
            sizeRatio = (float) getMeasuredWidth() / (float) imgWidth;
            int y = (int) (getMeasuredHeight() - sizeRatio * imgHeight) / 2;
            originPt.set(0, y);
        }

    }

    private Map<Integer, Paint> paintRandomPool = new HashMap<>();

    public ResultMaskView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        textPaint = new Paint();
        textPaint.setTextSize(30);
        textPaint.setARGB(255, 255, 255, 255);
    }

    private Paint getRandomMaskPaint(int index) {
        if (paintRandomPool.containsKey(index)) {

            return paintRandomPool.get(index);
        }

        int[] seed = new int[3];
        int offset = index % 3;
        seed[offset] = 255;

        Paint paint = new Paint();
        Random rnd = new Random();
        paint.setARGB(170,
                (rnd.nextInt(255) + seed[0]) / 2,
                (rnd.nextInt(255) + seed[1]) / 2,
                (rnd.nextInt(255) + seed[2]) / 2);
        paint.setStyle(Paint.Style.FILL_AND_STROKE);

        paintRandomPool.put(index, paint);

        return paint;
    }

    private Paint getFixColorPaint(int colorId) {

        // float alpha = 90;
        String[] colors = {"#FF0000", "#FF0000", "#00FF00", "#0000FF", "#FF00FF"};
        int index = colorId % colors.length;
        if (paintFixPool.containsKey(index)) {
            return paintFixPool.get(index);
        }

        int color = Color.parseColor(colors[index]);
        Paint paint = new Paint();
        paint.setColor(color);
        //paint.setAlpha(90);
        paintFixPool.put(index, paint);
        return paint;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        // 实时识别的时候第一次渲染
        if (mResultModelList == null) {
            super.onDraw(canvas);
            return;
        }

        preCaculate();

        int stokeWidth = 5;

        int fontSize = 38;
        int labelPadding = 5;
        int labelHeight = 46 + 2 * labelPadding;
        Paint paint = new Paint();
        paint.setColor(Color.parseColor("#3B85F5"));
        paint.setStrokeWidth(stokeWidth);
        paint.setStyle(Paint.Style.STROKE);

        Paint paintFillAlpha = new Paint();
        paintFillAlpha.setStyle(Paint.Style.FILL);
        paintFillAlpha.setColor(Color.parseColor("#3B85F5"));
        paintFillAlpha.setAlpha(50);

        Paint paintFill = new Paint();
        paintFill.setStyle(Paint.Style.FILL);
        paintFill.setColor(Color.parseColor("#3B85F5"));

        Paint paintText = new Paint();
        paintText.setColor(Color.WHITE);
        paintText.setTextAlign(Paint.Align.LEFT);
        paintText.setTextSize(fontSize);
        DecimalFormat df = new DecimalFormat("0.00");


        List<Float> points;
        for (int i = 0; i < mResultModelList.size(); i++) {

            BasePolygonResultModel model = mResultModelList.get(i);
            Path path = new Path();
            List<Point> polygon = model.getBounds(sizeRatio, originPt);

            path.moveTo(polygon.get(0).x, polygon.get(0).y);
            for (int j = 1; j < polygon.size(); j++) {
                path.lineTo(polygon.get(j).x, polygon.get(j).y);
            }
            path.close();

            if (model.isDrawPoints()) {
                Paint paintFillPoint = new Paint();
                paintFillPoint.setStyle(Paint.Style.FILL_AND_STROKE);

                paintFillPoint.setColor(Color.YELLOW);
                List<Point> polygonPoints = model.getBounds(sizeRatio, originPt);
                for (Point p : polygonPoints) {
                    canvas.drawCircle(p.x, p.y, 10 * sizeRatio, paintFillPoint);
                }

            }

            // 绘制框
            if (!model.isHasMask()) {
                if (model instanceof PoseViewResultModel && model.isHasGroupColor()) {
                    paint = getFixColorPaint(model.getColorId());
                    paint.setStrokeWidth(stokeWidth);
                    paint.setStyle(Paint.Style.STROKE);
                    paint.setAlpha(90);
                }
                canvas.drawPath(path, paint);
                canvas.drawPath(path, paintFillAlpha);
                if (model.isRect()) {
                    Rect rect = model.getRect(sizeRatio, originPt);
                    canvas.drawRect(new Rect(rect.left, rect.top, rect.right,
                            rect.top + labelHeight), paintFill);
                }
            }
            if (model.isRect()) {
                Rect rect = model.getRect(sizeRatio, originPt);
                canvas.drawText(model.getName() + " " + StringUtil.formatFloatString(model.getConfidence()),
                        rect.left + labelPadding,
                        rect.top + fontSize + labelPadding, paintText);
            }

            if (model.isTextOverlay()) {
                canvas.drawText(model.getName(),
                        polygon.get(0).x, polygon.get(0).y, textPaint);
            }

            // 绘制mask
            if (model.isHasMask()) {
                if (!model.isSemanticMask()) {
                    // 实例分割绘制
                    Paint paintMask = getRandomMaskPaint(model.getColorId());
                    points = new ArrayList<>();
                    byte[] maskData = model.getMask();

                    for (int w = 0; w < imgWidth * sizeRatio; w++) {
                        for (int h = 0; h < imgHeight * sizeRatio; h++) {

                            int realX = (int) (w / sizeRatio);
                            int realY = (int) (h / sizeRatio);

                            int offset = imgWidth * realY + realX;
                            if (offset < maskData.length && maskData[offset] == 1) {
                                points.add(originPt.x + (float) w);
                                points.add(originPt.y + (float) h);
                            }
                        }
                    }

                    float[] ptft = new float[points.size()];
                    for (int j = 0; j < points.size(); j++) {
                        ptft[j] = points.get(j);
                    }
                    canvas.drawPoints(ptft, paintMask);
                } else {
                    // 语义分割绘制
                    drawSemanticMask(canvas, model);
                }
            }
        }


        super.onDraw(canvas);
    }

    /**
     * 语义分割基于mask不同值绘制颜色
     */
    private void drawSemanticMask(Canvas canvas, BasePolygonResultModel model) {
        List<ColorPointsPair> colorPointsPairList = new ArrayList<>();
        for (int[] rgb : COLOR_MAP_DEF) {
            Paint paint = new Paint();
            paint.setARGB(170, rgb[0], rgb[1], rgb[2]);
            colorPointsPairList.add(new ColorPointsPair(paint));
        }

        byte[] maskData = model.getMask();
        for (int w = 0; w < imgWidth * sizeRatio; w++) {
            for (int h = 0; h < imgHeight * sizeRatio; h++) {

                int realX = (int) (w / sizeRatio);
                int realY = (int) (h / sizeRatio);

                int offset = imgWidth * realY + realX;
                byte label = maskData[offset];

                if (label >= colorPointsPairList.size()) {
                    for (int i = colorPointsPairList.size(); i <= label; i++ ) {
                        Paint newPaint = getRandomMaskPaint(i);
                        colorPointsPairList.add(new ColorPointsPair(newPaint));
                    }
                }

                colorPointsPairList.get(label).addPoint(originPt.x + (float) w);
                colorPointsPairList.get(label).addPoint(originPt.y + (float) h);
            }
        }

        for (ColorPointsPair pair : colorPointsPairList) {
            float[] points = pair.getPointsArray();
            if (points != null) {
                canvas.drawPoints(points, pair.getPaint());
            }
        }
    }

    /**
     * 用于分割模型，一个颜色对应一系列点
     */
    private class ColorPointsPair {
        private final Paint paint;
        private List<Float> points;

        public ColorPointsPair(Paint paint) {
            this.paint = paint;
        }

        public void addPoint(float point) {
            if (points == null) {
                points = new ArrayList<>();
            }
            points.add(point);
        }

        public float[] getPointsArray() {
            if (points == null || points.size() == 0) {
                return null;
            }
            float[] array = new float[points.size()];
            for (int i = 0; i < points.size(); i++) {
                array[i] = points.get(i);
            }
            return array;
        }

        public Paint getPaint() {
            return paint;
        }
    }
}
