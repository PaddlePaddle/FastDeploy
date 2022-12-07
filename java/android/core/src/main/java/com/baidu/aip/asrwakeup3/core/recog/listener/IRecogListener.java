package com.baidu.aip.asrwakeup3.core.recog.listener;

import com.baidu.aip.asrwakeup3.core.recog.RecogResult;

/**
 * 与SDK中回调参数的对应关系定义在RecogEventAdapter类中
 */

public interface IRecogListener {

    /**
     * CALLBACK_EVENT_ASR_READY
     * ASR_START 输入事件调用后，引擎准备完毕
     */
    void onAsrReady();

    /**
     * CALLBACK_EVENT_ASR_BEGIN
     * onAsrReady后检查到用户开始说话
     */
    void onAsrBegin();

    /**
     * CALLBACK_EVENT_ASR_END
     * 检查到用户开始说话停止，或者ASR_STOP 输入事件调用后，
     */
    void onAsrEnd();

    /**
     * CALLBACK_EVENT_ASR_PARTIAL resultType=partial_result
     * onAsrBegin 后 随着用户的说话，返回的临时结果
     *
     * @param results     可能返回多个结果，请取第一个结果
     * @param recogResult 完整的结果
     */
    void onAsrPartialResult(String[] results, RecogResult recogResult);

    /**
     * 语音的在线语义结果
     *
     * CALLBACK_EVENT_ASR_PARTIAL resultType=nlu_result
     * @param nluResult
     */
    void onAsrOnlineNluResult(String nluResult);

    /**
     *  不开启长语音仅回调一次，长语音的每一句话都会回调一次
     * CALLBACK_EVENT_ASR_PARTIAL resultType=final_result
     * 最终的识别结果
     *
     * @param results     可能返回多个结果，请取第一个结果
     * @param recogResult 完整的结果
     */
    void onAsrFinalResult(String[] results, RecogResult recogResult);

    /**
     * CALLBACK_EVENT_ASR_FINISH
     * @param recogResult 结束识别
     */
    void onAsrFinish(RecogResult recogResult);

    /**
     * CALLBACK_EVENT_ASR_FINISH error!=0
     *
     * @param errorCode
     * @param subErrorCode
     * @param descMessage
     * @param recogResult
     */
    void onAsrFinishError(int errorCode, int subErrorCode, String descMessage,
                          RecogResult recogResult);

    /**
     * 长语音识别结束
     */
    void onAsrLongFinish();

    /**
     * CALLBACK_EVENT_ASR_VOLUME
     * 音量回调
     *
     * @param volumePercent 音量的相对值，百分比，0-100
     * @param volume 音量绝对值
     */
    void onAsrVolume(int volumePercent, int volume);

    /**
     * CALLBACK_EVENT_ASR_AUDIO
     * @param data pcm格式，16bits 16000采样率
     *
     * @param offset
     * @param length
     */
    void onAsrAudio(byte[] data, int offset, int length);

    /**
     * CALLBACK_EVENT_ASR_EXIT
     * 引擎完成整个识别，空闲中
     */
    void onAsrExit();

    /**
     * CALLBACK_EVENT_ASR_LOADED
     * 离线命令词资源加载成功
     */
    void onOfflineLoaded();

    /**
     * CALLBACK_EVENT_ASR_UNLOADED
     * 离线命令词资源释放成功
     */
    void onOfflineUnLoaded();
}
