package com.baidu.paddle.fastdeploy.vision;

public class SuperResolutionResult {
    public byte[] mPixels;
    public long[] mShape;
    public boolean mInitialized = false;
    public long mCxxBuffer = 0;
    public boolean mEnableCxxBuffer = false;

    public SuperResolutionResult() {
        mInitialized = false;
    }
}
