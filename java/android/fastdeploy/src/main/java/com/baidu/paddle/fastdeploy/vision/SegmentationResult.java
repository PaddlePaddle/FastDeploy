package com.baidu.paddle.fastdeploy.vision;

import android.support.annotation.NonNull;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;

public class SegmentationResult {
    // Init from native
    public byte[] mLabelMap;
    public float[] mScoreMap;
    public long[] mShape;
    public boolean mContainScoreMap = false;
    public boolean mInitialized = false;
    // Cxx result context, some users may want to use
    // result pointer from native directly to boost
    // the performance of segmentation.
    public long mCxxBuffer = 0;
    public boolean mEnableCxxBuffer = false;

    public SegmentationResult() {
        mInitialized = false;
    }

    public boolean initialized() {
        return mInitialized;
    }

    public void setCxxBufferFlag(boolean flag) {
        mEnableCxxBuffer = flag;
    }

    public boolean releaseCxxBuffer() {
        if (mCxxBuffer == 0 || !mEnableCxxBuffer) {
            return false;
        }
        return releaseCxxBufferNative();
    }

    public void setLabelMap(@NonNull byte[] labelMapBuffer) {
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

    private native boolean releaseCxxBufferNative();

    // Initializes at the beginning.
    static {
        FastDeployInitializer.init();
    }
}
