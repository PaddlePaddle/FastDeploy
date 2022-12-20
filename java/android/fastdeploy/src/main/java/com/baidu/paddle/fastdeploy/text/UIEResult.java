package com.baidu.paddle.fastdeploy.text;

import java.util.HashMap;

public class UIEResult {
    public long mStart;
    public long mEnd;
    public double mProbability;
    public String mText;
    public HashMap<String, UIEResult[]> mRelation;
    public boolean mInitialized = false;

    public UIEResult() {
        mInitialized = false;
    }
}
