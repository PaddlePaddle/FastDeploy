package com.baidu.paddle.fastdeploy.vision;

import android.support.annotation.NonNull;
import java.util.Arrays;

public class OCRResult {
    public int[][] mBoxes;  // [n,8]
    public String[] mText;  // [n]
    public float[] mRecScores;  // [n]
    public float[] mClsScores;  // [n]
    public int[] mClsLabels;  // [n]
    public boolean mInitialized = false;

    public OCRResult() {
        mInitialized = false;
    }

    public boolean initialized() {
        return mInitialized;
    }

    public void setBoxes(@NonNull int[] boxesBuffer) {
        int boxesNum = boxesBuffer.length / 8;
        if (boxesNum > 0) {
            mBoxes = new int[boxesNum][8];
            for (int i = 0; i < boxesNum; ++i) {
                mBoxes[i] = Arrays.copyOfRange(
                        boxesBuffer, i * 8, (i + 1) * 8);
            }
        }
    }

    public void setText(@NonNull String[] textBuffer) {
        if (textBuffer.length > 0) {
            mText = textBuffer.clone();
        }
    }

    public void setRecScores(@NonNull float[] recScoresBuffer) {
        if (recScoresBuffer.length > 0) {
            mRecScores = recScoresBuffer.clone();
        }
    }

    public void setClsScores(@NonNull float[] clsScoresBuffer) {
        if (clsScoresBuffer.length > 0) {
            mClsScores = clsScoresBuffer.clone();
        }
    }

    public void setClsLabels(@NonNull int[] clsLabelBuffer) {
        if (clsLabelBuffer.length > 0) {
            mClsLabels = clsLabelBuffer.clone();
        }
    }
}
