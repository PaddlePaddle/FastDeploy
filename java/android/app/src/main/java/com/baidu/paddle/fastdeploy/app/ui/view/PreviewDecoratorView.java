package com.baidu.paddle.fastdeploy.app.ui.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Point;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.view.View;

import com.baidu.paddle.fastdeploy.app.examples.R;


/**
 * Created by ruanshimin on 2018/5/7.
 */

public class PreviewDecoratorView extends View {

    static final String RUNNING_HINT_TEXT = "对准识别物体，自动识别物体";

    static final String STOP_HINT_TEXT = "已暂定实时识别，请点击下方按钮开启";

    private String hintText = STOP_HINT_TEXT;

    public PreviewDecoratorView(Context context) {
        super(context);
    }

    public PreviewDecoratorView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public PreviewDecoratorView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        setLayerType(View.LAYER_TYPE_SOFTWARE, null);
    }

    public void setStatus(boolean isRunning) {
        if (isRunning) {
            hintText = RUNNING_HINT_TEXT;
        } else {
            hintText = STOP_HINT_TEXT;
        }
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        int offset = 50;
        int offsetBottom = 170;
        int stokeWidth = 15;
        int extendOffset = stokeWidth / 2;
        int height = getHeight();
        int width = getWidth();
        int barSize = 40;


        canvas.save();

        Paint paintBar = new Paint();
        paintBar.setColor(Color.WHITE);
        paintBar.setStrokeWidth(stokeWidth);
        paintBar.setStyle(Paint.Style.STROKE);


        Point leftTop = new Point(offset, offset);
        Point rightTop = new Point(width - offset, offset);
        Point leftBottom = new Point(offset, height - offsetBottom);
        Point rightBottom = new Point(width - offset, height - offsetBottom);


        Path path = new Path();
        path.moveTo(leftTop.x, leftTop.y + barSize);
        path.lineTo(leftTop.x, leftTop.y);
        path.lineTo(leftTop.x + barSize, leftTop.y);

        path.moveTo(rightTop.x - barSize, leftTop.y);
        path.lineTo(rightTop.x, rightTop.y);
        path.lineTo(rightTop.x, rightTop.y + barSize);

        path.moveTo(leftBottom.x, leftBottom.y - barSize);
        path.lineTo(leftBottom.x, leftBottom.y);
        path.lineTo(leftBottom.x + barSize, leftBottom.y);

        path.moveTo(rightBottom.x - barSize, rightBottom.y);
        path.lineTo(rightBottom.x, rightBottom.y);
        path.lineTo(rightBottom.x, rightBottom.y - barSize);


        // 绘制半透明遮罩
        Rect rect = new Rect(0, 0, width, height);
        Paint paint = new Paint();
        paint.setColor(Color.BLACK);
        paint.setAlpha(50);
        canvas.drawRect(rect, paint);

        // 擦除可见部分半透明遮罩
        rect = new Rect(leftTop.x - extendOffset, leftTop.y - extendOffset,
                rightBottom.x + extendOffset, rightBottom.y + extendOffset);
        canvas.clipRect(rect);
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

        // 绘制四个小角
        canvas.drawPath(path, paintBar);
        canvas.restore();
        // 设置留给hint的位置
        rect = new Rect(0, rightBottom.y + extendOffset, width, height);

        drawHint(canvas, rect, hintText);

        super.onDraw(canvas);
    }

    public void drawHint(Canvas canvas, Rect rect, String text) {
        int textSizePx = 16;
        int imageSizePx = 48;
        int padInTextAndIcon = 22;
        Paint textPaint = new Paint();
        textPaint.setTextSize(textSizePx);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextAlign(Paint.Align.LEFT);

        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.scan_icon);

        float textWidth = getTextWidth(this.getContext(), text, textPaint, textSizePx);
        int width = padInTextAndIcon + imageSizePx + (int) textWidth;
        int height = imageSizePx;

        int startX = rect.left + (rect.width() - width) / 2;
        int startY = rect.top + (rect.height() - height) / 2;

        Rect srcRect = new Rect(0, 0, bitmap.getWidth(), bitmap.getHeight());
        Rect destRect = new Rect(startX, startY, startX + imageSizePx, startY + imageSizePx);

        Paint mBitPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        mBitPaint.setFilterBitmap(true);
        mBitPaint.setDither(true);
        canvas.drawBitmap(bitmap, srcRect, destRect, mBitPaint);

        startX += padInTextAndIcon + imageSizePx;

        canvas.drawText(text, startX, startY + textSizePx + 23, textPaint);
    }

    public float getTextWidth(Context context, String text, Paint paint, int textSize) {
        float scaledDensity = getResources().getDisplayMetrics().scaledDensity;
        paint.setTextSize(scaledDensity * textSize);
        return paint.measureText(text);
    }
}
