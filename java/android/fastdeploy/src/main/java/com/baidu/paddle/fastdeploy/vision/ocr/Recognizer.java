package com.baidu.paddle.fastdeploy.vision.ocr;

import com.baidu.paddle.fastdeploy.RuntimeOption;

public class Recognizer {
    // TODO(qiuyanjun): Allows users to set model parameters,
    //  such as rec_img_h, rec_img_w, rec_image_shape, etc.
    //  These parameters should be passed in via JNI.
    public String mModelFile;
    public String mParamsFile;
    public String mLabelPath;
    public RuntimeOption mRuntimeOption;
    public boolean mInitialized = false;

    public Recognizer() {
        mModelFile = "";
        mParamsFile = "";
        mLabelPath = "";
        mRuntimeOption = new RuntimeOption();
        mInitialized = false;
    }

    public Recognizer(String modelFile,
                      String paramsFile,
                      String labelPath) {
        mModelFile = modelFile;
        mParamsFile = paramsFile;
        mLabelPath = labelPath;
        mRuntimeOption = new RuntimeOption();
        mInitialized = true;
    }

    public Recognizer(String modelFile,
                      String paramsFile,
                      String labelPath,
                      RuntimeOption option) {
        mModelFile = modelFile;
        mParamsFile = paramsFile;
        mLabelPath = labelPath;
        mRuntimeOption = option;
        mInitialized = true;
    }

    public boolean initialized() {
        return mInitialized;
    }
}
