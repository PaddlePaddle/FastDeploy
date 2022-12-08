package com.baidu.paddle.fastdeploy.app.examples.ernie.applications.core.wakeup.listener;

import com.baidu.paddle.fastdeploy.app.examples.ernie.applications.core.wakeup.WakeUpResult;

/**
 * Created by fujiayi on 2017/6/21.
 */

public interface IWakeupListener {


    void onSuccess(String word, WakeUpResult result);

    void onStop();

    void onError(int errorCode, String errorMessge, WakeUpResult result);

    void onASrAudio(byte[] data, int offset, int length);
}
