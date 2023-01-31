package com.baidu.paddle.fastdeploy.vision;

import android.graphics.Bitmap;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;


public class Visualize {
    // visClassification, visDetection, visSegmentation, visOcr, ...
    // Visualize DetectionResult without labels
    public static boolean visDetection(Bitmap ARGB8888Bitmap,
                                       DetectionResult result) {
        return visDetectionNative(
                ARGB8888Bitmap, result,
                0.f, 2, 0.5f,
                new String[]{});
    }

    public static boolean visDetection(Bitmap ARGB8888Bitmap,
                                       DetectionResult result,
                                       float scoreThreshold) {
        return visDetectionNative(
                ARGB8888Bitmap,
                result,
                scoreThreshold,
                2,
                0.5f,
                new String[]{});
    }

    public static boolean visDetection(Bitmap ARGB8888Bitmap,
                                       DetectionResult result,
                                       float scoreThreshold,
                                       int lineSize,
                                       float fontSize) {
        return visDetectionNative(
                ARGB8888Bitmap,
                result,
                scoreThreshold,
                lineSize,
                fontSize,
                new String[]{});
    }

    // Visualize DetectionResult with labels
    public static boolean visDetection(Bitmap ARGB8888Bitmap,
                                       DetectionResult result,
                                       String[] labels) {
        return visDetectionNative(
                ARGB8888Bitmap,
                result,
                0.f, 2, 0.5f,
                labels);
    }

    public static boolean visDetection(Bitmap ARGB8888Bitmap,
                                       DetectionResult result,
                                       String[] labels,
                                       float scoreThreshold,
                                       int lineSize,
                                       float fontSize) {
        return visDetectionNative(
                ARGB8888Bitmap,
                result,
                scoreThreshold,
                lineSize,
                fontSize,
                labels);
    }

    // Visualize ClassifyResult without labels
    public static boolean visClassification(Bitmap ARGB8888Bitmap,
                                            ClassifyResult result) {
        return visClassificationNative(
                ARGB8888Bitmap, result,
                0.f, 1,
                new String[]{});

    }

    public static boolean visClassification(Bitmap ARGB8888Bitmap,
                                            ClassifyResult result,
                                            float scoreThreshold,
                                            float fontSize) {
        return visClassificationNative(
                ARGB8888Bitmap, result,
                scoreThreshold,
                fontSize,
                new String[]{});

    }

    // Visualize ClassifyResult with labels
    public static boolean visClassification(Bitmap ARGB8888Bitmap,
                                            ClassifyResult result,
                                            String[] labels) {
        return visClassificationNative(
                ARGB8888Bitmap, result,
                0.f, 1,
                labels);

    }

    public static boolean visClassification(Bitmap ARGB8888Bitmap,
                                            ClassifyResult result,
                                            String[] labels,
                                            float scoreThreshold,
                                            float fontSize) {
        return visClassificationNative(
                ARGB8888Bitmap,
                result,
                scoreThreshold,
                fontSize,
                labels);

    }

    // Visualize OCRResult
    public static boolean visOcr(Bitmap ARGB8888Bitmap,
                                 OCRResult result) {
        return visOcrNative(
                ARGB8888Bitmap,
                result);
    }

    // Visualize SegmentationResult
    public static boolean visSegmentation(Bitmap ARGB8888Bitmap,
                                          SegmentationResult result) {
        return visSegmentationNative(
                ARGB8888Bitmap,
                result,
                0.5f);
    }

    public static boolean visSegmentation(Bitmap ARGB8888Bitmap,
                                          SegmentationResult result,
                                          float weight) {
        return visSegmentationNative(
                ARGB8888Bitmap,
                result,
                weight);
    }

    // Visualize FaceDetectionResult
    public static boolean visFaceDetection(Bitmap ARGB8888Bitmap,
                                           FaceDetectionResult result) {
        return visFaceDetectionNative(
                ARGB8888Bitmap,
                result,
                2, 0.5f);
    }

    public static boolean visFaceDetection(Bitmap ARGB8888Bitmap,
                                           FaceDetectionResult result,
                                           int lineSize,
                                           float fontSize) {
        return visFaceDetectionNative(
                ARGB8888Bitmap,
                result,
                lineSize,
                fontSize);
    }

    // Visualize KeyPointDetectionResult
    public static boolean visKeypointDetection(Bitmap ARGB8888Bitmap,
                                               KeyPointDetectionResult result) {
        return visKeyPointDetectionNative(
                ARGB8888Bitmap,
                result,
                0.5f);
    }

    public static boolean visKeypointDetection(Bitmap ARGB8888Bitmap,
                                               KeyPointDetectionResult result,
                                               float confThreshold) {
        return visKeyPointDetectionNative(
                ARGB8888Bitmap,
                result,
                confThreshold);
    }

    // VisDetection in native
    private static native boolean visDetectionNative(Bitmap ARGB8888Bitmap,
                                                     DetectionResult result,
                                                     float scoreThreshold,
                                                     int lineSize,
                                                     float fontSize,
                                                     String[] labels);

    // VisClassification in native
    private static native boolean visClassificationNative(Bitmap ARGB8888Bitmap,
                                                          ClassifyResult result,
                                                          float scoreThreshold,
                                                          float fontSize,
                                                          String[] labels);

    // VisOcr in native
    private static native boolean visOcrNative(Bitmap ARGB8888Bitmap,
                                               OCRResult result);

    // VisSegmentation in native
    private static native boolean visSegmentationNative(Bitmap ARGB8888Bitmap,
                                                        SegmentationResult result,
                                                        float weight);

    // VisFaceDetection in native
    private static native boolean visFaceDetectionNative(Bitmap ARGB8888Bitmap,
                                                         FaceDetectionResult result,
                                                         int lineSize,
                                                         float fontSize);

    // VisKeypointDetection in native
    private static native boolean visKeyPointDetectionNative(Bitmap ARGB8888Bitmap,
                                                             KeyPointDetectionResult result,
                                                             float confThreshold);

    /* Initializes at the beginning */
    static {
        FastDeployInitializer.init();
    }
}
