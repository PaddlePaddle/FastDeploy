package com.baidu.paddle.fastdeploy;

public class RuntimeOption {
    public int mCpuThreadNum = 1;
    public boolean mEnableLiteFp16 = false;
    public boolean mEnableRecordTimeOfRuntime = false;
    public LitePowerMode mLitePowerMode = LitePowerMode.LITE_POWER_NO_BIND;
    public String mLiteOptimizedModelDir = "";

    public RuntimeOption() {
        mCpuThreadNum = 1;
        mEnableLiteFp16 = false;
        mEnableRecordTimeOfRuntime = false;
        mLitePowerMode = LitePowerMode.LITE_POWER_NO_BIND;
        mLiteOptimizedModelDir = "";
    }

    public void enableLiteFp16() {
        mEnableLiteFp16 = true;
    }

    public void disableLiteFP16() {
        mEnableLiteFp16 = false;
    }

    public void setCpuThreadNum(int threadNum) {
        mCpuThreadNum = threadNum;
    }

    public void setLitePowerMode(LitePowerMode mode) {
        mLitePowerMode = mode;
    }

    public void setLitePowerMode(String modeStr) {
        mLitePowerMode = parseLitePowerModeFromString(modeStr);
    }

    public void setLiteOptimizedModelDir(String modelDir) {
        mLiteOptimizedModelDir = modelDir;
    }

    public void enableRecordTimeOfRuntime() {
        mEnableRecordTimeOfRuntime = true;
    }

    // Helpers: parse lite power mode from string
    public static LitePowerMode parseLitePowerModeFromString(String modeStr) {
        if (modeStr.equalsIgnoreCase("LITE_POWER_HIGH")) {
            return LitePowerMode.LITE_POWER_HIGH;
        } else if (modeStr.equalsIgnoreCase("LITE_POWER_LOW")) {
            return LitePowerMode.LITE_POWER_LOW;
        } else if (modeStr.equalsIgnoreCase("LITE_POWER_FULL")) {
            return LitePowerMode.LITE_POWER_FULL;
        } else if (modeStr.equalsIgnoreCase("LITE_POWER_NO_BIND")) {
            return LitePowerMode.LITE_POWER_NO_BIND;
        } else if (modeStr.equalsIgnoreCase("LITE_POWER_RAND_HIGH")) {
            return LitePowerMode.LITE_POWER_RAND_HIGH;
        } else if (modeStr.equalsIgnoreCase("LITE_POWER_RAND_LOW")) {
            return LitePowerMode.LITE_POWER_RAND_LOW;
        } else {
            return LitePowerMode.LITE_POWER_NO_BIND;
        }
    }
}
