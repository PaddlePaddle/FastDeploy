package com.baidu.paddle.fastdeploy.pipeline;

import android.graphics.Bitmap;

import com.baidu.paddle.fastdeploy.FastDeployInitializer;
import com.baidu.paddle.fastdeploy.vision.OCRResult;
import com.baidu.paddle.fastdeploy.vision.ocr.DBDetector;
import com.baidu.paddle.fastdeploy.vision.ocr.Classifier;
import com.baidu.paddle.fastdeploy.vision.ocr.Recognizer;
import com.baidu.paddle.fastdeploy.RuntimeOption;

public class PPOCRBase {
    protected long mNativeHandlerContext = 0; // Context from native.
    protected boolean mInitialized = false;

    public PPOCRBase() {
        mInitialized = false;
    }

    // Constructor w/o classifier
    public PPOCRBase(DBDetector detModel,
                     Recognizer recModel,
                     PPOCRVersion OCRVersionTag) {
        init_(detModel, new Classifier(), recModel, OCRVersionTag);
    }

    public PPOCRBase(DBDetector detModel,
                     Classifier clsModel,
                     Recognizer recModel,
                     PPOCRVersion OCRVersionTag) {
        init_(detModel, clsModel, recModel, OCRVersionTag);
    }

    // Call init manually w/o classifier
    public boolean init(DBDetector detModel,
                        Recognizer recModel,
                        PPOCRVersion OCRVersionTag) {
        return init_(detModel, new Classifier(), recModel, OCRVersionTag);
    }

    public boolean init(DBDetector detModel,
                        Classifier clsModel,
                        Recognizer recModel,
                        PPOCRVersion OCRVersionTag) {
        return init_(detModel, clsModel, recModel, OCRVersionTag);
    }

    public boolean release() {
        mInitialized = false;
        if (mNativeHandlerContext == 0) {
            return false;
        }
        return releaseNative(mNativeHandlerContext);
    }

    public boolean initialized() {
        return mInitialized;
    }

    // Predict without image saving and bitmap rendering.
    public OCRResult predict(Bitmap ARGB8888Bitmap) {
        if (mNativeHandlerContext == 0) {
            return new OCRResult();
        }
        // Only support ARGB8888 bitmap in native now.
        OCRResult result = predictNative(mNativeHandlerContext, ARGB8888Bitmap,
                false, "", false);
        if (result == null) {
            return new OCRResult();
        }
        return result;
    }

    public OCRResult predict(Bitmap ARGB8888Bitmap, boolean rendering) {
        if (mNativeHandlerContext == 0) {
            return new OCRResult();
        }
        // Only support ARGB8888 bitmap in native now.
        OCRResult result = predictNative(mNativeHandlerContext, ARGB8888Bitmap,
                false, "", rendering);
        if (result == null) {
            return new OCRResult();
        }
        return result;
    }

    // Predict with image saving and bitmap rendering (will cost more times)
    public OCRResult predict(Bitmap ARGB8888Bitmap,
                             String savedImagePath) {
        // scoreThreshold is for visualizing only.
        if (mNativeHandlerContext == 0) {
            return new OCRResult();
        }
        // Only support ARGB8888 bitmap in native now.
        OCRResult result = predictNative(mNativeHandlerContext, ARGB8888Bitmap,
                true, savedImagePath, true);
        if (result == null) {
            return new OCRResult();
        }
        return result;
    }

    public boolean init_(DBDetector detModel,
                         Classifier clsModel,
                         Recognizer recModel,
                         PPOCRVersion OCRVersionTag) {
        if (!mInitialized) {
            mNativeHandlerContext = bindNative(
                    OCRVersionTag.ordinal(),
                    detModel.mModelFile,
                    detModel.mParamsFile,
                    clsModel.mModelFile,
                    clsModel.mParamsFile,
                    recModel.mModelFile,
                    recModel.mParamsFile,
                    recModel.mLabelPath,
                    detModel.mRuntimeOption,
                    clsModel.mRuntimeOption,
                    recModel.mRuntimeOption,
                    clsModel.initialized());
            if (mNativeHandlerContext != 0) {
                mInitialized = true;
            }
            return mInitialized;
        } else {
            // release current native context and bind a new one.
            if (release()) {
                mNativeHandlerContext = bindNative(
                        OCRVersionTag.ordinal(),
                        detModel.mModelFile,
                        detModel.mParamsFile,
                        clsModel.mModelFile,
                        clsModel.mParamsFile,
                        recModel.mModelFile,
                        recModel.mParamsFile,
                        recModel.mLabelPath,
                        detModel.mRuntimeOption,
                        clsModel.mRuntimeOption,
                        recModel.mRuntimeOption,
                        clsModel.initialized());
                if (mNativeHandlerContext != 0) {
                    mInitialized = true;
                }
                return mInitialized;
            }
            return false;
        }
    }

    // Bind predictor from native context.
    private native long bindNative(int PPOCRVersionTag,
                                   String detModelFile,
                                   String detParamsFile,
                                   String clsModelFile,
                                   String clsParamsFile,
                                   String recModelFile,
                                   String recParamsFile,
                                   String recLabelPath,
                                   RuntimeOption detRuntimeOption,
                                   RuntimeOption clsRuntimeOption,
                                   RuntimeOption recRuntimeOption,
                                   boolean haveClsModel);

    // Call prediction from native context with rendering.
    private native OCRResult predictNative(long nativeHandlerContext,
                                           Bitmap ARGB8888Bitmap,
                                           boolean saved,
                                           String savedImagePath,
                                           boolean rendering);

    // Release buffers allocated in native context.
    private native boolean releaseNative(long nativeHandlerContext);

    // Initializes at the beginning.
    static {
        FastDeployInitializer.init();
    }

}
