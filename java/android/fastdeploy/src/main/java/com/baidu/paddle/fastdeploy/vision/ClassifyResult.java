package com.baidu.paddle.fastdeploy.vision;

import android.support.annotation.NonNull;

public class ClassifyResult {
    public float[] mScores;  // [n]
    public int[] mLabelIds;  // [n]
    public boolean mInitialized = false;

    public ClassifyResult() {
        mInitialized = false;
    }

    public boolean initialized() {
        return mInitialized;
    }

    public void setScores(@NonNull float[] scoresBuffer) {
        if (scoresBuffer.length > 0) {
            mScores = scoresBuffer.clone();
        }
    }

    public void setLabelIds(@NonNull int[] labelIdsBuffer) {
        if (labelIdsBuffer.length > 0) {
            mLabelIds = labelIdsBuffer.clone();
        }
    }
}
