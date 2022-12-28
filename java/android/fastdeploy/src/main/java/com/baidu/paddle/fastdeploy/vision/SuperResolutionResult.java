package com.baidu.paddle.fastdeploy.vision;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;

public class SuperResolutionResult {
    public byte[] mPixels; // RGB pixels
    public long[] mShape; // (H,W)
    public boolean mInitialized = false;
    public long mCxxBuffer = 0;
    public boolean mEnableCxxBuffer = false;

    public SuperResolutionResult() {
        mInitialized = false;
    }

    public boolean initialized() {
        return mInitialized;
    }

    public void setCxxBufferFlag(boolean flag) {
        mEnableCxxBuffer = flag;
    }

    public boolean releaseCxxBuffer() {
        if (mCxxBuffer == 0 || !mEnableCxxBuffer) {
            return false;
        }
        return releaseCxxBufferNative();
    }

    private native boolean releaseCxxBufferNative();

    // Initializes at the beginning.
    static {
        FastDeployInitializer.init();
    }
}
