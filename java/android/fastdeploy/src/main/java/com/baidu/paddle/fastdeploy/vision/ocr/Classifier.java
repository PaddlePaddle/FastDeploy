package com.baidu.paddle.fastdeploy.vision.ocr;

import com.baidu.paddle.fastdeploy.RuntimeOption;

public class Classifier {
    // TODO(qiuyanjun): Allows users to set model parameters,
    //  such as cls_thresh, cls_image_shape, is_scale, etc.
    //  These parameters should be passed in via JNI.
    public String mModelFile;
    public String mParamsFile;
    public RuntimeOption mRuntimeOption;
    public boolean mInitialized = false;

    public Classifier() {
        mModelFile = "";
        mParamsFile = "";
        mRuntimeOption = new RuntimeOption();
        mInitialized = false;
    }

    public Classifier(String modelFile,
                      String paramsFile) {
        mModelFile = modelFile;
        mParamsFile = paramsFile;
        mRuntimeOption = new RuntimeOption();
        mInitialized = true;
    }

    public Classifier(String modelFile,
                      String paramsFile,
                      RuntimeOption option) {
        mModelFile = modelFile;
        mParamsFile = paramsFile;
        mRuntimeOption = option;
        mInitialized = true;
    }

    public boolean initialized() {
        return mInitialized;
    }
}
