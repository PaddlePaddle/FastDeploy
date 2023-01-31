package com.baidu.paddle.fastdeploy.vision;

import android.support.annotation.NonNull;

import java.util.Arrays;

public class DetectionResult {
    public float[][] mBoxes; // [n,4]
    public float[] mScores;  // [n]
    public int[] mLabelIds;  // [n]
    public boolean mInitialized = false;

    public DetectionResult() {
        mInitialized = false;
    }

    public boolean initialized() {
        return mInitialized;
    }

    public void setBoxes(@NonNull float[] boxesBuffer) {
        int boxesNum = boxesBuffer.length / 4;
        if (boxesNum > 0) {
            mBoxes = new float[boxesNum][4];
            for (int i = 0; i < boxesNum; ++i) {
                mBoxes[i] = Arrays.copyOfRange(
                        boxesBuffer, i * 4, (i + 1) * 4);
            }
        }
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
