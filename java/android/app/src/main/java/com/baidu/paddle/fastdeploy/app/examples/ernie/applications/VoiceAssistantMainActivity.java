package com.baidu.paddle.fastdeploy.app.examples.ernie.applications;

import static com.baidu.paddle.fastdeploy.ui.Utils.isNetworkAvailable;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.baidu.aip.asrwakeup3.core.mini.AutoCheck;
import com.baidu.aip.asrwakeup3.core.util.AuthUtil;
import com.baidu.paddle.fastdeploy.app.examples.R;
import com.baidu.speech.EventListener;
import com.baidu.speech.EventManager;
import com.baidu.speech.EventManagerFactory;
import com.baidu.speech.asr.SpeechConstant;

import org.json.JSONObject;

import java.util.LinkedHashMap;
import java.util.Map;

public class VoiceAssistantMainActivity extends Activity implements View.OnClickListener, EventListener {
    private Button startVoiceBtn;
    private TextView voiceOutput;
    private Button startIntentBtn;
    private TextView intentOutput;
    private ImageView back;
    private EventManager asr;
    private Boolean isStartVoice = false;
    protected boolean enableOffline = false;
    private String voiceTxt = "";

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Fullscreen
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.voice_assistant_activity_main);

        init();
    }

    private void init() {
        asr = EventManagerFactory.create(this, "asr");
        asr.registerListener(this);
        startVoiceBtn = findViewById(R.id.btn_voice);
        startVoiceBtn.setOnClickListener(this);
        voiceOutput = findViewById(R.id.tv_voice_output);
        back = findViewById(R.id.iv_back);
        back.setOnClickListener(this);
        startIntentBtn = findViewById(R.id.btn_intent);
        startIntentBtn.setOnClickListener(this);
        intentOutput = findViewById(R.id.tv_intent_output);
        if (enableOffline) {
            loadOfflineEngine(); // 测试离线命令词请开启, 测试 ASR_OFFLINE_ENGINE_GRAMMER_FILE_PATH 参数时开启
        }
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.btn_voice:
                if (!isNetworkAvailable(this)) {
                    new AlertDialog.Builder(VoiceAssistantMainActivity.this)
                            .setMessage("请先连接互联网。")
                            .setCancelable(true)
                            .show();
                    return;
                }
                if (!isStartVoice) {
                    isStartVoice = true;
                    startVoiceBtn.setText("停止录音");
                    start();
                } else {
                    isStartVoice = false;
                    startVoiceBtn.setText("开始录音");
                    stop();
                }
                break;
            case R.id.iv_back:
                finish();
                break;
            case R.id.btn_intent:
                if (voiceTxt.equals("")) {
                    new AlertDialog.Builder(VoiceAssistantMainActivity.this)
                            .setMessage("请先录音。")
                            .setCancelable(true)
                            .show();
                    return;
                }
                intentOutput.setText("我刚才说了：" + voiceTxt);
                break;
        }
    }

    @Override
    public void onEvent(String name, String params, byte[] data, int offset, int length) {
        if (name.equals(SpeechConstant.CALLBACK_EVENT_ASR_PARTIAL)) {
            if (params.contains("\"final_result\"")) {
                if (params.contains("[")) {
                    voiceTxt = params.substring(params.lastIndexOf('[') + 1, params.lastIndexOf(']'));
                }
                voiceOutput.setText(voiceTxt);
            }
        }
    }

    private void start() {
        Map<String, Object> params = AuthUtil.getParam();
        String event = null;
        event = SpeechConstant.ASR_START;
        if (enableOffline) {
            params.put(SpeechConstant.DECODER, 2);
        }
        params.put(SpeechConstant.ACCEPT_AUDIO_VOLUME, false);
        (new AutoCheck(getApplicationContext(), new Handler() {
            public void handleMessage(Message msg) {
                if (msg.what == 100) {
                    AutoCheck autoCheck = (AutoCheck) msg.obj;
                    synchronized (autoCheck) {
                        String message = autoCheck.obtainErrorMessage();
                        Log.e(getClass().getName(), message);
                    }
                }
            }
        }, enableOffline)).checkAsr(params);
        String json = null;
        json = new JSONObject(params).toString();
        asr.send(event, json, null, 0, 0);
    }

    /**
     * enableOffline设为true时，在onCreate中调用
     * 基于SDK离线命令词1.4 加载离线资源(离线时使用)
     */
    private void loadOfflineEngine() {
        Map<String, Object> params = new LinkedHashMap<String, Object>();
        params.put(SpeechConstant.DECODER, 2);
        params.put(SpeechConstant.ASR_OFFLINE_ENGINE_GRAMMER_FILE_PATH, "assets://baidu_speech_grammar.bsg");
        asr.send(SpeechConstant.ASR_KWS_LOAD_ENGINE, new JSONObject(params).toString(), null, 0, 0);
    }

    /**
     * enableOffline为true时，在onDestory中调用，与loadOfflineEngine对应
     * 基于SDK集成5.1 卸载离线资源步骤(离线时使用)
     */
    private void unloadOfflineEngine() {
        asr.send(SpeechConstant.ASR_KWS_UNLOAD_ENGINE, null, null, 0, 0);
    }

    private void stop() {
        asr.send(SpeechConstant.ASR_STOP, null, null, 0, 0);
    }

    @Override
    protected void onPause() {
        super.onPause();
        asr.send(SpeechConstant.ASR_CANCEL, "{}", null, 0, 0);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        asr.send(SpeechConstant.ASR_CANCEL, "{}", null, 0, 0);
        if (enableOffline) {
            unloadOfflineEngine(); // 测试离线命令词请开启, 测试 ASR_OFFLINE_ENGINE_GRAMMER_FILE_PATH 参数时开启
        }
        asr.unregisterListener(this);
    }
}
