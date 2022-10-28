package com.baidu.paddle.fastdeploy.vision;

import android.graphics.Bitmap;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;


public class Visualize {
    // TODO(qiuyanjun):
    //  VisClassification, VisSegmentation, VisMatting, VisOcr, ...

    // Visualize DetectionResult without labels
    public static boolean visDetection(Bitmap ARGB8888Bitmap,
                                       DetectionResult result) {
        return visDetectionNative(
                ARGB8888Bitmap,
                result.mBoxes,
                result.mScores,
                result.mLabelIds,
                0.f, 1, 0.5f,
                new String[]{});
    }

    public static boolean visDetection(Bitmap ARGB8888Bitmap,
                                       DetectionResult result,
                                       float score_threshold,
                                       int line_size,
                                       float font_size) {
        return visDetectionNative(
                ARGB8888Bitmap,
                result.mBoxes,
                result.mScores,
                result.mLabelIds,
                score_threshold,
                line_size,
                font_size,
                new String[]{});
    }

    // Visualize DetectionResult with labels
    public static boolean visDetection(Bitmap ARGB8888Bitmap,
                                       DetectionResult result,
                                       String[] labels) {
        return visDetectionNative(
                ARGB8888Bitmap,
                result.mBoxes,
                result.mScores,
                result.mLabelIds,
                0.f, 1, 0.5f,
                labels);
    }

    public static boolean visDetection(Bitmap ARGB8888Bitmap,
                                       DetectionResult result,
                                       float score_threshold,
                                       int line_size,
                                       float font_size,
                                       String[] labels) {
        return visDetectionNative(
                ARGB8888Bitmap,
                result.mBoxes,
                result.mScores,
                result.mLabelIds,
                score_threshold,
                line_size,
                font_size,
                labels);
    }

    // VisDetection in native
    public static native boolean visDetectionNative(Bitmap ARGB8888Bitmap,
                                                    float[][] boxes,
                                                    float[] scores,
                                                    int[] labelIds,
                                                    float score_threshold,
                                                    int line_size,
                                                    float font_size,
                                                    String[] labels);


    /* Initializes at the beginning */
    static {
        FastDeployInitializer.init();
    }
}
