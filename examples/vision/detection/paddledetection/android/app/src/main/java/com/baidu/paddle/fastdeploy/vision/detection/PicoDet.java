package com.baidu.paddle.fastdeploy.vision.detection;

import android.graphics.Bitmap;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.vision.DetectionResult;

public class PicoDet {
    protected long mNativeModelContext = 0; // Context from native.
    protected boolean mInitialized = false;

    public PicoDet() {
        mInitialized = false;
    }

    // Constructor without label file
    public PicoDet(String modelFile,
                   String paramsFile,
                   String configFile,
                   RuntimeOption option) {
        init_(modelFile, paramsFile, configFile, "", option);
    }

    // Constructor with label file
    public PicoDet(String modelFile,
                   String paramsFile,
                   String configFile,
                   String labelFile,
                   RuntimeOption option) {
        init_(modelFile, paramsFile, configFile, labelFile, option);
    }

    // Call init manually without label file
    public boolean init(String modelFile,
                        String paramsFile,
                        String configFile,
                        RuntimeOption option) {
        return init_(modelFile, paramsFile, configFile, "", option);
    }

    // Call init manually with label file
    public boolean init(String modelFile,
                        String paramsFile,
                        String configFile,
                        String labelFile,
                        RuntimeOption option) {
        return init_(modelFile, paramsFile, configFile, labelFile, option);
    }

    public boolean release() {
        mInitialized = false;
        if (mNativeModelContext == 0) {
            return false;
        }
        return releaseNative(mNativeModelContext);
    }

    public boolean initialized() {
        return mInitialized;
    }

    // Predict without image saving and bitmap rendering.
    public DetectionResult predict(Bitmap ARGB8888Bitmap) {
        if (mNativeModelContext == 0) {
            return new DetectionResult();
        }
        // Only support ARGB8888 bitmap in native now.
        return new DetectionResult(predictNative(
                mNativeModelContext, ARGB8888Bitmap, false,
                "", 0.f, false));
    }

    // Predict with image saving and bitmap rendering (will cost more times)
    public DetectionResult predict(Bitmap ARGB8888Bitmap,
                                   String savedImagePath,
                                   float scoreThreshold) {
        // scoreThreshold is for visualizing only.
        if (mNativeModelContext == 0) {
            return new DetectionResult();
        }
        // Only support ARGB8888 bitmap in native now.
        return new DetectionResult(predictNative(
                mNativeModelContext, ARGB8888Bitmap, true,
                savedImagePath, scoreThreshold, true));
    }


    private boolean init_(String modelFile,
                          String paramsFile,
                          String configFile,
                          String labelFile,
                          RuntimeOption option) {
        if (!mInitialized) {
            mNativeModelContext = bindNative(
                    modelFile,
                    paramsFile,
                    configFile,
                    option.mCpuThreadNum,
                    option.mEnableLiteFp16,
                    option.mLitePowerMode.ordinal(),
                    option.mLiteOptimizedModelDir,
                    option.mEnableRecordTimeOfRuntime, labelFile);
            if (mNativeModelContext != 0) {
                mInitialized = true;
            }
            return mInitialized;
        } else {
            // release current native context and bind a new one.
            if (release()) {
                mNativeModelContext = bindNative(
                        modelFile,
                        paramsFile,
                        configFile,
                        option.mCpuThreadNum,
                        option.mEnableLiteFp16,
                        option.mLitePowerMode.ordinal(),
                        option.mLiteOptimizedModelDir,
                        option.mEnableRecordTimeOfRuntime, labelFile);
                if (mNativeModelContext != 0) {
                    mInitialized = true;
                }
                return mInitialized;
            }
            return false;
        }
    }

    // Bind predictor from native context.
    private static native long bindNative(String modelFile,
                                          String paramsFile,
                                          String configFile,
                                          int cpuNumThread,
                                          boolean enableLiteFp16,
                                          int litePowerMode,
                                          String liteOptimizedModelDir,
                                          boolean enableRecordTimeOfRuntime,
                                          String labelFile);

    // Call prediction from native context.
    private static native long predictNative(long nativeModelContext,
                                             Bitmap ARGB8888Bitmap,
                                             boolean saved,
                                             String savedImagePath,
                                             float scoreThreshold,
                                             boolean rendering);

    // Release buffers allocated in native context.
    private static native boolean releaseNative(long nativeModelContext);

    // Initializes at the beginning.
    static {
        FastDeployInitializer.init();
    }
}

