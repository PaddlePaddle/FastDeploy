package com.baidu.paddle.fastdeploy.vision.classification;

import android.graphics.Bitmap;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.vision.ClassifyResult;

public class PaddleClasModel {
    protected long mNativeModelContext = 0; // Context from native.
    protected boolean mInitialized = false;

    public PaddleClasModel() {
        mInitialized = false;
    }

    // Constructor with default runtimeOption
    public PaddleClasModel(String modelFile,
                           String paramsFile,
                           String configFile) {
        init_(modelFile, paramsFile, configFile, "", new RuntimeOption());
    }

    public PaddleClasModel(String modelFile,
                           String paramsFile,
                           String configFile,
                           String labelFile) {
        init_(modelFile, paramsFile, configFile, labelFile, new RuntimeOption());
    }

    // Constructor without label file
    public PaddleClasModel(String modelFile,
                           String paramsFile,
                           String configFile,
                           RuntimeOption runtimeOption) {
        init_(modelFile, paramsFile, configFile, "", runtimeOption);
    }

    // Constructor with label file
    public PaddleClasModel(String modelFile,
                           String paramsFile,
                           String configFile,
                           String labelFile,
                           RuntimeOption runtimeOption) {
        init_(modelFile, paramsFile, configFile, labelFile, runtimeOption);
    }

    // Call init manually without label file
    public boolean init(String modelFile,
                        String paramsFile,
                        String configFile,
                        RuntimeOption runtimeOption) {
        return init_(modelFile, paramsFile, configFile, "", runtimeOption);
    }

    // Call init manually with label file
    public boolean init(String modelFile,
                        String paramsFile,
                        String configFile,
                        String labelFile,
                        RuntimeOption runtimeOption) {
        return init_(modelFile, paramsFile, configFile, labelFile, runtimeOption);
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
    public ClassifyResult predict(Bitmap ARGB8888Bitmap) {
        if (mNativeModelContext == 0) {
            return new ClassifyResult();
        }
        // Only support ARGB8888 bitmap in native now.
        ClassifyResult result = predictNative(mNativeModelContext, ARGB8888Bitmap,
                false, "", 0.f, false);
        if (result == null) {
            return new ClassifyResult();
        }
        return result;
    }

    public ClassifyResult predict(Bitmap ARGB8888Bitmap,
                                  boolean rendering,
                                  float scoreThreshold) {
        if (mNativeModelContext == 0) {
            return new ClassifyResult();
        }
        // Only support ARGB8888 bitmap in native now.
        ClassifyResult result = predictNative(mNativeModelContext, ARGB8888Bitmap,
                false, "", scoreThreshold, rendering);
        if (result == null) {
            return new ClassifyResult();
        }
        return result;
    }

    // Predict with image saving and bitmap rendering (will cost more times)
    public ClassifyResult predict(Bitmap ARGB8888Bitmap,
                                  String savedImagePath,
                                  float scoreThreshold) {
        // scoreThreshold is for visualizing only.
        if (mNativeModelContext == 0) {
            return new ClassifyResult();
        }
        // Only support ARGB8888 bitmap in native now.
        ClassifyResult result = predictNative(
                mNativeModelContext, ARGB8888Bitmap,
                true, savedImagePath, scoreThreshold, true);
        if (result == null) {
            return new ClassifyResult();
        }
        return result;
    }

    // Internal init_ method
    private boolean init_(String modelFile,
                          String paramsFile,
                          String configFile,
                          String labelFile,
                          RuntimeOption runtimeOption) {
        if (!mInitialized) {
            mNativeModelContext = bindNative(
                    modelFile,
                    paramsFile,
                    configFile,
                    runtimeOption,
                    labelFile);
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
                        runtimeOption,
                        labelFile);
                if (mNativeModelContext != 0) {
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
                                   RuntimeOption runtimeOption,
                                   String labelFile);

    // Call prediction from native context with rendering.
    private native ClassifyResult predictNative(long nativeModelContext,
                                                Bitmap ARGB8888Bitmap,
                                                boolean saved,
                                                String savedImagePath,
                                                float scoreThreshold,
                                                boolean rendering);

    // Release buffers allocated in native context.
    private native boolean releaseNative(long nativeModelContext);

    // Initializes at the beginning.
    static {
        FastDeployInitializer.init();
    }

}
