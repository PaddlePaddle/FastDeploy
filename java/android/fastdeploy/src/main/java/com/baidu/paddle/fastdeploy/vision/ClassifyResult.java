package com.baidu.paddle.fastdeploy.vision;

import android.support.annotation.NonNull;

public class ClassifyResult {
    public float[] mScores;  // [n]
    public int[] mLabelIds;  // [n]
    public boolean mInitialized = false;

    public ClassifyResult() {
        mInitialized = false;
    }

    public ClassifyResult(long nativeResultContext) {
        mInitialized = copyAllFromNativeContext(nativeResultContext);
    }

    public boolean initialized() {
        return mInitialized;
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

    private boolean copyAllFromNativeContext(long nativeResultContext) {
        if (nativeResultContext == 0) {
            return false;
        }
        setScores(copyScoresFromNative(nativeResultContext));
        setLabelIds(copyLabelIdsFromNative(nativeResultContext));
        // WARN: must release ctx.
        return releaseNative(nativeResultContext);
    }

    // Fetch native buffers from native context.
    private native float[] copyScoresFromNative(long nativeResultContext);

    private native int[] copyLabelIdsFromNative(long nativeResultContext);

    private native boolean releaseNative(long nativeResultContext);

}
