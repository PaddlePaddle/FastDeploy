package com.baidu.paddle.fastdeploy.app.examples.ernie.applications.core.wakeup.listener;

import android.os.Handler;

import com.baidu.paddle.fastdeploy.app.examples.ernie.applications.core.recog.IStatus;
import com.baidu.paddle.fastdeploy.app.examples.ernie.applications.core.wakeup.WakeUpResult;

/**
 * Created by fujiayi on 2017/9/21.
 */

public class RecogWakeupListener extends SimpleWakeupListener implements IStatus {

    private static final String TAG = "RecogWakeupListener";

    private Handler handler;

    public RecogWakeupListener(Handler handler) {
        this.handler = handler;
    }

    @Override
    public void onSuccess(String word, WakeUpResult result) {
        super.onSuccess(word, result);
        handler.sendMessage(handler.obtainMessage(STATUS_WAKEUP_SUCCESS));
    }
}
