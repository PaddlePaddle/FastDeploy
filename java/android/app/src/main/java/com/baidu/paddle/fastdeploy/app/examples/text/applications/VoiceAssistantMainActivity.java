package com.baidu.paddle.fastdeploy.app.examples.text.applications;

import static com.baidu.paddle.fastdeploy.ui.Utils.isNetworkAvailable;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class VoiceAssistantMainActivity extends Activity implements View.OnClickListener, EventListener {
    private Button startVoiceBtn;
    private TextView voiceOutput;
    private Button startIntentBtn;
    private TextView intentOutput;
    private ImageButton btnSettings;
    private ImageView back;
    private EventManager asr;
    private Boolean isStartVoice = false;
    private String voiceTxt = "";
    private int times = 0;
    private final int REQUEST_PERMISSION = 0;

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
        checkPermission();
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
        btnSettings = findViewById(R.id.btn_settings);
        btnSettings.setOnClickListener(this);
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
            case R.id.btn_settings:
                startActivity(new Intent(VoiceAssistantMainActivity.this, VoiceAssistantSettingsActivity.class));
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
        }, false)).checkAsr(params);
        String json = null;
        json = new JSONObject(params).toString();
        asr.send(event, json, null, 0, 0);
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
        asr.unregisterListener(this);
    }

    private void checkPermission() {
        times++;
        final List<String> permissionsList = new ArrayList<>();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if ((checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED))
                permissionsList.add(Manifest.permission.RECORD_AUDIO);
            if ((checkSelfPermission(Manifest.permission.ACCESS_NETWORK_STATE) != PackageManager.PERMISSION_GRANTED))
                permissionsList.add(Manifest.permission.ACCESS_NETWORK_STATE);
            if ((checkSelfPermission(Manifest.permission.INTERNET) != PackageManager.PERMISSION_GRANTED)) {
                permissionsList.add(Manifest.permission.INTERNET);
            }
            if ((checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED)) {
                permissionsList.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
            }
            if (permissionsList.size() != 0) {
                if (times == 1) {
                    requestPermissions(permissionsList.toArray(new String[permissionsList.size()]),
                            REQUEST_PERMISSION);
                } else {
                    new AlertDialog.Builder(this)
                            .setCancelable(true)
                            .setTitle("提示")
                            .setMessage("获取不到授权，APP将无法正常使用，请允许APP获取权限！")
                            .setPositiveButton("确定", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface arg0, int arg1) {
                                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                                        requestPermissions(permissionsList.toArray(new String[permissionsList.size()]),
                                                REQUEST_PERMISSION);
                                    }
                                }
                            }).setNegativeButton("取消", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface arg0, int arg1) {
                                    finish();
                                }
                            }).show();
                }
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        checkPermission();
    }
}
