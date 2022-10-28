package com.baidu.paddle.fastdeploy.pipeline;

import com.baidu.paddle.fastdeploy.vision.ocr.Classifier;
import com.baidu.paddle.fastdeploy.vision.ocr.DBDetector;
import com.baidu.paddle.fastdeploy.vision.ocr.Recognizer;

public class PPOCRv2 extends PPOCRBase {
    public PPOCRv2() {
        super();
    }

    // Constructor w/o classifier
    public PPOCRv2(DBDetector detModel,
                   Recognizer recModel) {
        super(detModel, recModel, PPOCRVersion.OCR_V2);
    }

    public PPOCRv2(DBDetector detModel,
                   Classifier clsModel,
                   Recognizer recModel) {
        super(detModel, clsModel, recModel, PPOCRVersion.OCR_V2);
    }

    // Call init manually w/o classifier
    public boolean init(DBDetector detModel,
                        Recognizer recModel) {
        return init(detModel, recModel, PPOCRVersion.OCR_V2);
    }

    public boolean init(DBDetector detModel,
                        Classifier clsModel,
                        Recognizer recModel) {
        return init(detModel, clsModel, recModel, PPOCRVersion.OCR_V2);
    }
}


























