package com.baidu.paddle.fastdeploy.vision.keypointdetection;

import android.graphics.Bitmap;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.vision.KeyPointDetectionResult;

public class PPTinyPose {
    protected long mCxxContext = 0; // Context from native.
    protected boolean mUseDark = true;
    protected boolean mInitialized = false;

    public PPTinyPose() {
        mInitialized = false;
    }

    // Constructor with default runtime option
    public PPTinyPose(String modelFile,
                      String paramsFile,
                      String configFile) {
        init_(modelFile, paramsFile, configFile, new RuntimeOption());
    }

    // Constructor without label file
    public PPTinyPose(String modelFile,
                      String paramsFile,
                      String configFile,
                      RuntimeOption runtimeOption) {
        init_(modelFile, paramsFile, configFile, runtimeOption);
    }

    public void setUseDark(boolean flag) {
        mUseDark = flag;
    }

    // Call init manually without label file
    public boolean init(String modelFile,
                        String paramsFile,
                        String configFile,
                        RuntimeOption runtimeOption) {
        return init_(modelFile, paramsFile, configFile, runtimeOption);
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
    public KeyPointDetectionResult predict(Bitmap ARGB8888Bitmap) {
        if (mCxxContext == 0) {
            return new KeyPointDetectionResult();
        }
        // Only support ARGB8888 bitmap in native now.
        KeyPointDetectionResult result = predictNative(mCxxContext, ARGB8888Bitmap,
                false, "", false, 0.f);
        if (result == null) {
            return new KeyPointDetectionResult();
        }
        return result;
    }

    public KeyPointDetectionResult predict(Bitmap ARGB8888Bitmap,
                                           boolean rendering,
                                           float confThreshold) {
        if (mCxxContext == 0) {
            return new KeyPointDetectionResult();
        }
        // Only support ARGB8888 bitmap in native now.
        KeyPointDetectionResult result = predictNative(mCxxContext, ARGB8888Bitmap,
                false, "", rendering, confThreshold);
        if (result == null) {
            return new KeyPointDetectionResult();
        }
        return result;
    }

    // Predict with image saving and bitmap rendering (will cost more times)
    public KeyPointDetectionResult predict(Bitmap ARGB8888Bitmap,
                                           String savedImagePath,
                                           float confThreshold) {
        // confThreshold is for visualizing only.
        if (mCxxContext == 0) {
            return new KeyPointDetectionResult();
        }
        // Only support ARGB8888 bitmap in native now.
        KeyPointDetectionResult result = predictNative(
                mCxxContext, ARGB8888Bitmap, true,
                savedImagePath, true, confThreshold);
        if (result == null) {
            return new KeyPointDetectionResult();
        }
        return result;
    }

    private boolean init_(String modelFile,
                          String paramsFile,
                          String configFile,
                          RuntimeOption runtimeOption) {
        if (!mInitialized) {
            mCxxContext = bindNative(
                    modelFile,
                    paramsFile,
                    configFile,
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
                        configFile,
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
                                   String configFile,
                                   RuntimeOption runtimeOption);

    // Call prediction from native context with rendering.
    private native KeyPointDetectionResult predictNative(long CxxContext,
                                                         Bitmap ARGB8888Bitmap,
                                                         boolean saveImage,
                                                         String savePath,
                                                         boolean rendering,
                                                         float confThreshold);

    // Release buffers allocated in native context.
    private native boolean releaseNative(long CxxContext);

    // Initializes at the beginning.
    static {
        FastDeployInitializer.init();
    }
}
