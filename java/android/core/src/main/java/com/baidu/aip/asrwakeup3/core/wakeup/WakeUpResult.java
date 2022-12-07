package com.baidu.aip.asrwakeup3.core.wakeup;

import com.baidu.aip.asrwakeup3.core.util.MyLogger;
import com.baidu.speech.asr.SpeechConstant;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * Created by fujiayi on 2017/6/24.
 */
public class WakeUpResult {
    private String name;
    private String origalJson;
    private String word;
    private String desc;
    private int errorCode;

    private static int ERROR_NONE = 0;

    private static final String TAG = "WakeUpResult";

    public boolean hasError() {
        return errorCode != ERROR_NONE;
    }

    public String getOrigalJson() {
        return origalJson;
    }

    public void setOrigalJson(String origalJson) {
        this.origalJson = origalJson;
    }

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }

    public String getDesc() {
        return desc;
    }

    public void setDesc(String desc) {
        this.desc = desc;
    }

    public int getErrorCode() {
        return errorCode;
    }

    public void setErrorCode(int errorCode) {
        this.errorCode = errorCode;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public static WakeUpResult parseJson(String name, String jsonStr) {
        WakeUpResult result = new WakeUpResult();
        result.setOrigalJson(jsonStr);
        try {
            JSONObject json = new JSONObject(jsonStr);
            if (SpeechConstant.CALLBACK_EVENT_WAKEUP_SUCCESS.equals(name)) {
                int error = json.optInt("errorCode");
                result.setErrorCode(error);
                result.setDesc(json.optString("errorDesc"));
                if (!result.hasError()) {
                    result.setWord(json.optString("word"));
                }
            } else {
                int error = json.optInt("error");
                result.setErrorCode(error);
                result.setDesc(json.optString("desc"));
            }

        } catch (JSONException e) {
            MyLogger.error(TAG, "Json parse error" + jsonStr);
            e.printStackTrace();
        }

        return result;
    }
}
