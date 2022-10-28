package com.baidu.paddle.fastdeploy.pipeline;

import com.baidu.paddle.fastdeploy.vision.ocr.Classifier;
import com.baidu.paddle.fastdeploy.vision.ocr.DBDetector;
import com.baidu.paddle.fastdeploy.vision.ocr.Recognizer;

public class PPOCRv3 extends PPOCRBase {
    public PPOCRv3() {
        super();
    }

    // Constructor w/o classifier
    public PPOCRv3(DBDetector detModel,
                   Recognizer recModel) {
        super(detModel, recModel, PPOCRVersion.OCR_V3);
    }

    public PPOCRv3(DBDetector detModel,
                   Classifier clsModel,
                   Recognizer recModel) {
        super(detModel, clsModel, recModel, PPOCRVersion.OCR_V3);
    }

    // Call init manually w/o classifier
    public boolean init(DBDetector detModel,
                        Recognizer recModel) {
        return init(detModel, recModel, PPOCRVersion.OCR_V3);
    }

    public boolean init(DBDetector detModel,
                        Classifier clsModel,
                        Recognizer recModel) {
        return init(detModel, clsModel, recModel, PPOCRVersion.OCR_V3);
    }
}

