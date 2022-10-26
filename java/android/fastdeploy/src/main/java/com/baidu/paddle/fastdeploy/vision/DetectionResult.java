package com.baidu.paddle.fastdeploy.vision;

import android.support.annotation.NonNull;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;

import java.util.Arrays;

public class DetectionResult {
    // Not support MaskRCNN now.
    public float[][] mBoxes; // [n,4]
    public float[] mScores;  // [n]
    public int[] mLabelIds;  // [n]
    public boolean mInitialized = false;

    public DetectionResult() {
        mInitialized = false;
    }

    public DetectionResult(long nativeResultContext) {
        mInitialized = copyAllFromNativeContext(nativeResultContext);
    }

    public boolean initialized() {
        return mInitialized;
    }

    // Setup results from native buffers.
    private boolean copyAllFromNativeContext(long nativeResultContext) {
        if (nativeResultContext == 0) {
            return false;
        }
        if (copyBoxesNumFromNative(nativeResultContext) > 0) {
            setBoxes(copyBoxesFromNative(nativeResultContext));
            setScores(copyScoresFromNative(nativeResultContext));
            setLabelIds(copyLabelIdsFromNative(nativeResultContext));
        }
        // WARN: must release ctx.
        return releaseNative(nativeResultContext);
    }

    private void setBoxes(@NonNull float[] boxesBuffer) {
        int boxesNum = boxesBuffer.length / 4;
        if (boxesNum > 0) {
            mBoxes = new float[boxesNum][4];
            for (int i = 0; i < boxesNum; ++i) {
                mBoxes[i] = Arrays.copyOfRange(
                        boxesBuffer, i * 4, (i + 1) * 4);
            }
        }
    }

    private void setScores(@NonNull float[] scoresBuffer) {
        if (scoresBuffer.length > 0) {
            mScores = scoresBuffer.clone();
        }
    }

    private void setLabelIds(@NonNull int[] labelIdsBuffer) {
        if (labelIdsBuffer.length > 0) {
            mLabelIds = labelIdsBuffer.clone();
        }
    }

    // Fetch native buffers from native context.
    private static native int copyBoxesNumFromNative(long nativeResultContext);

    private static native float[] copyBoxesFromNative(long nativeResultContext);

    private static native float[] copyScoresFromNative(long nativeResultContext);

    private static native int[] copyLabelIdsFromNative(long nativeResultContext);

    private static native boolean releaseNative(long nativeResultContext);

    // Initializes at the beginning.
    static {
        FastDeployInitializer.init();
    }
}
