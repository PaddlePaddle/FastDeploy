package com.baidu.paddle.fastdeploy.vision;

import android.support.annotation.NonNull;

import java.util.Arrays;

public class KeyPointDetectionResult {
    public float[][] mKeyPoints; // [n*num_joints, 2]
    public float[] mScores;  // [n*num_joints]
    public int mNumJoints = -1;
    public boolean mInitialized = false;

    public KeyPointDetectionResult() {
        mInitialized = false;
    }

    public boolean initialized() {
        return mInitialized;
    }

    public void setKeyPoints(@NonNull float[] keyPointsBuffer) {
        int pointNum = keyPointsBuffer.length / 2;
        if (pointNum > 0) {
            mKeyPoints = new float[pointNum][2];
            for (int i = 0; i < pointNum; ++i) {
                mKeyPoints[i] = Arrays.copyOfRange(
                        keyPointsBuffer, i * 2, (i + 1) * 2);
            }
        }
    }

    public void setScores(@NonNull float[] scoresBuffer) {
        if (scoresBuffer.length > 0) {
            mScores = scoresBuffer.clone();
        }
    }

    public void setNumJoints(int numJoints) {
        mNumJoints = numJoints;
    }

}
