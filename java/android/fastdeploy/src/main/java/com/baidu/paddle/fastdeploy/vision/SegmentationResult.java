package com.baidu.paddle.fastdeploy.vision;

import android.support.annotation.NonNull;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;

public class SegmentationResult {
    // Init from native
    public int[] mLabelMap;
    public float[] mScoreMap;
    public long[] mShape;
    public boolean mContainScoreMap = false;
    public boolean mInitialized = false;

    public SegmentationResult() {
        mInitialized = false;
    }

    public boolean initialized() {
        return mInitialized;
    }

    public void setLabelMap(@NonNull int[] labelMapBuffer) {
        if (labelMapBuffer.length > 0) {
            mLabelMap = labelMapBuffer.clone();
        }
    }

    public void setScoreMap(@NonNull float[] scoreMapBuffer) {
        if (scoreMapBuffer.length > 0) {
            mScoreMap = scoreMapBuffer.clone();
        }
    }

    public void setShape(@NonNull long[] shapeBuffer) {
        if (shapeBuffer.length > 0) {
            mShape = shapeBuffer.clone();
        }
    }

    public void setContainScoreMap(boolean containScoreMap) {
        mContainScoreMap = containScoreMap;
    }
}
