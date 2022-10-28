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
                                       String[] labels,
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
                labels);
    }

    // Visualize ClassifyResult without labels
    public static boolean visClassification(Bitmap ARGB8888Bitmap,
                                            ClassifyResult result) {
        return visClassificationNative(
                ARGB8888Bitmap,
                result.mScores,
                result.mLabelIds,
                0.f, 1,
                new String[]{});

    }

    public static boolean visClassification(Bitmap ARGB8888Bitmap,
                                            ClassifyResult result,
                                            float score_threshold,
                                            float font_size) {
        return visClassificationNative(
                ARGB8888Bitmap,
                result.mScores,
                result.mLabelIds,
                score_threshold,
                font_size,
                new String[]{});

    }

    // Visualize ClassifyResult with labels
    public static boolean visClassification(Bitmap ARGB8888Bitmap,
                                            ClassifyResult result,
                                            String[] labels) {
        return visClassificationNative(
                ARGB8888Bitmap,
                result.mScores,
                result.mLabelIds,
                0.f, 1,
                labels);

    }

    public static boolean visClassification(Bitmap ARGB8888Bitmap,
                                            ClassifyResult result,
                                            String[] labels,
                                            float score_threshold,
                                            float font_size) {
        return visClassificationNative(
                ARGB8888Bitmap,
                result.mScores,
                result.mLabelIds,
                score_threshold,
                font_size,
                labels);

    }

    // Visualize OCRResult
    public static boolean visOcr(Bitmap ARGB8888Bitmap,
                                 OCRResult result) {
        return visOcrNative(
                ARGB8888Bitmap,
                result.mBoxes,
                result.mText,
                result.mRecScores,
                result.mClsScores,
                result.mClsLabels);
    }

    // VisDetection in native
    private static native boolean visDetectionNative(Bitmap ARGB8888Bitmap,
                                                     float[][] boxes,
                                                     float[] scores,
                                                     int[] labelIds,
                                                     float score_threshold,
                                                     int line_size,
                                                     float font_size,
                                                     String[] labels);

    // VisClassification in native
    private static native boolean visClassificationNative(Bitmap ARGB8888Bitmap,
                                                          float[] scores,
                                                          int[] labelIds,
                                                          float score_threshold,
                                                          float font_size,
                                                          String[] labels);

    // VisOCRResult in native
    private static native boolean visOcrNative(Bitmap ARGB8888Bitmap,
                                               int[][] boxes,
                                               String[] text,
                                               float[] recScores,
                                               float[] clsScores,
                                               int[] clsLabels);


    /* Initializes at the beginning */
    static {
        FastDeployInitializer.init();
    }
}
