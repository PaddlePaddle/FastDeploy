package com.baidu.paddle.fastdeploy.vision.ocr;

import com.baidu.paddle.fastdeploy.RuntimeOption;

public class DBDetector {
    // TODO(qiuyanjun): Allows users to set model parameters,
    //  such as det_db_box_thresh, det_db_thresh, use_dilation, etc.
    //  These parameters should be passed in via JNI.
    public String mModelFile;
    public String mParamsFile;
    public RuntimeOption mRuntimeOption;
    public boolean mInitialized = false;

    public DBDetector() {
        mModelFile = "";
        mParamsFile = "";
        mRuntimeOption = new RuntimeOption();
        mInitialized = false;
    }

    public DBDetector(String modelFile,
                      String paramsFile) {
        mModelFile = modelFile;
        mParamsFile = paramsFile;
        mRuntimeOption = new RuntimeOption();
        mInitialized = true;
    }

    public DBDetector(String modelFile,
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
