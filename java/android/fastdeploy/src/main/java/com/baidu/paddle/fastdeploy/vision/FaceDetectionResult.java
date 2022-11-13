package com.baidu.paddle.fastdeploy.vision;

import android.support.annotation.NonNull;

import java.util.Arrays;

public class FaceDetectionResult {
    public float[][] mBoxes; // [n,4]
    public float[] mScores;  // [n]
    public float[][] mLandmarks; // [n,2]
    int mLandmarksPerFace = 0;
    public boolean mInitialized = false;

    public FaceDetectionResult() {
        mInitialized = false;
        mLandmarksPerFace = 0;
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

    public void setLandmarks(@NonNull float[] landmarksBuffer) {
        int landmarksNum = landmarksBuffer.length / 2;
        if (landmarksNum > 0) {
            mLandmarks = new float[landmarksNum][2];
            for (int i = 0; i < landmarksNum; ++i) {
                mLandmarks[i] = Arrays.copyOfRange(
                        landmarksBuffer, i * 2, (i + 1) * 2);
            }
        }
    }

    public void setLandmarksPerFace(int landmarksPerFace) {
        mLandmarksPerFace = landmarksPerFace;
    }
}
