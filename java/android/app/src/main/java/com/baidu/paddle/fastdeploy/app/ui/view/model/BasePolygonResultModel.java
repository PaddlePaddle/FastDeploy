package com.baidu.paddle.fastdeploy.app.ui.view.model;

import android.graphics.Point;
import android.graphics.Rect;

import java.util.ArrayList;
import java.util.List;

public class BasePolygonResultModel extends BaseResultModel {
    private int colorId;
    private boolean isRect;
    private boolean isTextOverlay;

    private boolean isHasGroupColor = false;
    private boolean isDrawPoints = false;


    public boolean isTextOverlay() {
        return isTextOverlay;
    }

    public void setTextOverlay(boolean textOverlay) {
        isTextOverlay = textOverlay;
    }

    public int getColorId() {
        return colorId;
    }

    public void setColorId(int colorId) {
        this.colorId = colorId;
    }

    private byte[] mask;

    /**
     * 是否是语义分割mask
     */
    private boolean semanticMask;

    private Rect rect;

    BasePolygonResultModel() {
        super();
    }

    BasePolygonResultModel(int index, String name, float confidence, Rect bounds) {
        super(index, name, confidence);
        parseFromRect(bounds);
    }

    BasePolygonResultModel(int index, String name, float confidence, List<Point> bounds) {
        super(index, name, confidence);
        this.bounds = bounds;
    }

    public Rect getRect() {
        return rect;
    }

    public Rect getRect(float ratio, Point origin) {
        return new Rect((int) (origin.x + rect.left * ratio),
                (int) (origin.y + rect.top * ratio),
                (int) (origin.x + rect.right * ratio),
                (int) (origin.y + rect.bottom * ratio));
    }

    private void parseFromRect(Rect rect) {
        Point ptTL = new Point(rect.left, rect.top);
        Point ptTR = new Point(rect.right, rect.top);
        Point ptRB = new Point(rect.right, rect.bottom);
        Point ptLB = new Point(rect.left, rect.bottom);
        this.bounds = new ArrayList<>();
        this.bounds.add(ptTL);
        this.bounds.add(ptTR);
        this.bounds.add(ptRB);
        this.bounds.add(ptLB);
        this.rect = rect;
        isRect = true;
    }

    public boolean isRect() {
        return isRect;
    }

    public void setRect(boolean rect) {
        isRect = rect;
    }

    public byte[] getMask() {
        return mask;
    }

    public void setMask(byte[] mask) {
        this.mask = mask;
    }

    public void setSemanticMask(boolean semanticMask) {
        this.semanticMask = semanticMask;
    }

    public boolean isSemanticMask() {
        return semanticMask;
    }

    private List<Point> bounds;

    public List<Point> getBounds() {
        return bounds;
    }

    public List<Point> getBounds(float ratio, Point origin) {
        List<Point> pointList = new ArrayList<>();
        for (Point pt : bounds) {
            int nx = (int) (origin.x + pt.x * ratio);
            int ny = (int) (origin.y + pt.y * ratio);
            pointList.add(new Point(nx, ny));
        }
        return pointList;
    }

    public void setBounds(List<Point> bounds) {
        this.bounds = bounds;
    }

    public void setBounds(Rect bounds) {
        parseFromRect(bounds);
    }

    public boolean isHasMask() {
        return (mask != null);
    }

    public boolean isHasGroupColor() {
        return isHasGroupColor;
    }

    public void setHasGroupColor(boolean hasGroupColor) {
        isHasGroupColor = hasGroupColor;
    }

    public boolean isDrawPoints() {
        return isDrawPoints;
    }

    public void setDrawPoints(boolean drawPoints) {
        isDrawPoints = drawPoints;
    }
}
