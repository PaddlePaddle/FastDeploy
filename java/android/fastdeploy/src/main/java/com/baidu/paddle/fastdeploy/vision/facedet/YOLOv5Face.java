package com.baidu.paddle.fastdeploy.vision.facedet;

import android.graphics.Bitmap;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.vision.FaceDetectionResult;

public class YOLOv5Face {
    public int[] mSize = {320, 320};
    protected long mCxxContext = 0; // Context from native.
    protected boolean mInitialized = false;

    public YOLOv5Face() {
        mInitialized = false;
    }

    // Constructor with default runtime option
    public YOLOv5Face(String modelFile,
                      String paramsFile) {
        init_(modelFile, paramsFile, new RuntimeOption());
    }

    // Constructor with custom runtime option
    public YOLOv5Face(String modelFile,
                      String paramsFile,
                      RuntimeOption runtimeOption) {
        init_(modelFile, paramsFile, runtimeOption);
    }

    public boolean init(String modelFile,
                        String paramsFile,
                        RuntimeOption runtimeOption) {
        return init_(modelFile, paramsFile, runtimeOption);
    }

    public boolean release() {
        mInitialized = false;
        if (mCxxContext == 0) {
            return false;
        }
        return releaseNative(mCxxContext);
    }

    public boolean initialized() {
        return mInitialized;
    }

    // Predict without image saving and bitmap rendering.
    public FaceDetectionResult predict(Bitmap ARGB8888Bitmap) {
        if (mCxxContext == 0) {
            return new FaceDetectionResult();
        }
        // Only support ARGB8888 bitmap in native now.
        FaceDetectionResult result = predictNative(mCxxContext, ARGB8888Bitmap,
                0.25f, 0.4f, false, "", false);
        if (result == null) {
            return new FaceDetectionResult();
        }
        return result;
    }

    public FaceDetectionResult predict(Bitmap ARGB8888Bitmap,
                                       float confThreshold,
                                       float nmsIouThreshold) {
        if (mCxxContext == 0) {
            return new FaceDetectionResult();
        }
        // Only support ARGB8888 bitmap in native now.
        FaceDetectionResult result = predictNative(mCxxContext, ARGB8888Bitmap,
                confThreshold, nmsIouThreshold, false, "", false);
        if (result == null) {
            return new FaceDetectionResult();
        }
        return result;
    }

    public FaceDetectionResult predict(Bitmap ARGB8888Bitmap,
                                       boolean rendering,
                                       float confThreshold,
                                       float nmsIouThreshold) {
        if (mCxxContext == 0) {
            return new FaceDetectionResult();
        }
        // Only support ARGB8888 bitmap in native now.
        FaceDetectionResult result = predictNative(mCxxContext, ARGB8888Bitmap,
                confThreshold, nmsIouThreshold, false, "", rendering);
        if (result == null) {
            return new FaceDetectionResult();
        }
        return result;
    }

    // Predict with image saving and bitmap rendering (will cost more times)
    public FaceDetectionResult predict(Bitmap ARGB8888Bitmap,
                                       String savedImagePath,
                                       float confThreshold,
                                       float nmsIouThreshold) {
        // scoreThreshold is for visualizing only.
        if (mCxxContext == 0) {
            return new FaceDetectionResult();
        }
        // Only support ARGB8888 bitmap in native now.
        FaceDetectionResult result = predictNative(
                mCxxContext, ARGB8888Bitmap, confThreshold, nmsIouThreshold,
                true, savedImagePath, true);
        if (result == null) {
            return new FaceDetectionResult();
        }
        return result;
    }

    private boolean init_(String modelFile,
                          String paramsFile,
                          RuntimeOption runtimeOption) {
        if (!mInitialized) {
            mCxxContext = bindNative(
                    modelFile,
                    paramsFile,
                    runtimeOption);
            if (mCxxContext != 0) {
                mInitialized = true;
            }
            return mInitialized;
        } else {
            // release current native context and bind a new one.
            if (release()) {
                mCxxContext = bindNative(
                        modelFile,
                        paramsFile,
                        runtimeOption);
                if (mCxxContext != 0) {
                    mInitialized = true;
                }
                return mInitialized;
            }
            return false;
        }
    }

    // Bind predictor from native context.
    private native long bindNative(String modelFile,
                                   String paramsFile,
                                   RuntimeOption runtimeOption);

    // Call prediction from native context with rendering.
    private native FaceDetectionResult predictNative(long CxxContext,
                                                     Bitmap ARGB8888Bitmap,
                                                     float confThreshold,
                                                     float nmsIouThreshold,
                                                     boolean saveImage,
                                                     String savePath,
                                                     boolean rendering);

    // Release buffers allocated in native context.
    private native boolean releaseNative(long CxxContext);

    // Initializes at the beginning.
    static {
        FastDeployInitializer.init();
    }
}
