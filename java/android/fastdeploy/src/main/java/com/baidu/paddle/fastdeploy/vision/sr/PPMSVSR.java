package com.baidu.paddle.fastdeploy.vision.sr;

import android.graphics.Bitmap;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.vision.SuperResolutionResult;

public class PPMSVSR {
    protected long mCxxContext = 0; // Context from native.
    protected boolean mInitialized = false;

    public PPMSVSR() {
        mInitialized = false;
    }

    // Constructor with default runtime option
    public PPMSVSR(String modelFile,
                   String paramsFile,
                   String configFile) {
        init_(modelFile, paramsFile, configFile, new RuntimeOption());
    }

    // Constructor with custom runtime option
    public PPMSVSR(String modelFile,
                   String paramsFile,
                   String configFile,
                   RuntimeOption runtimeOption) {
        init_(modelFile, paramsFile, configFile, runtimeOption);
    }

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

    public SuperResolutionResult predict(Bitmap ARGB8888Bitmap) {
        if (mCxxContext == 0) {
            return new SuperResolutionResult();
        }
        return predictNative(mCxxContext, ARGB8888Bitmap,
                false, "");
    }

    public SuperResolutionResult predict(Bitmap ARGB8888Bitmap,
                                         String savedImagePath) {
        if (mCxxContext == 0) {
            return new SuperResolutionResult();
        }
        return predictNative(mCxxContext, ARGB8888Bitmap,
                true, savedImagePath);
    }

    public boolean predict(Bitmap ARGB8888Bitmap,
                           SuperResolutionResult result) {
        if (mCxxContext == 0) {
            return false;
        }
        return predictNativeV2(mCxxContext, ARGB8888Bitmap,
                result, false, "");
    }

    public boolean predict(Bitmap ARGB8888Bitmap,
                           SuperResolutionResult result,
                           String savedImagePath) {
        if (mCxxContext == 0) {
            return false;
        }
        return predictNativeV2(mCxxContext, ARGB8888Bitmap,
                result, true, savedImagePath);
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

    private native SuperResolutionResult predictNative(long CxxContext,
                                                       Bitmap ARGB8888Bitmap,
                                                       boolean saveImage,
                                                       String savePath);

    private native boolean predictNativeV2(long CxxContext,
                                           Bitmap ARGB8888Bitmap,
                                           SuperResolutionResult result,
                                           boolean saveImage,
                                           String savePath);

    // Release buffers allocated in native context.
    private native boolean releaseNative(long CxxContext);

    // Initializes at the beginning.
    static {
        FastDeployInitializer.init();
    }
}
