package com.baidu.aip.asrwakeup3.core.mini;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.v4.content.ContextCompat;
import android.util.Log;

import com.baidu.speech.asr.SpeechConstant;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeSet;

import javax.net.ssl.HttpsURLConnection;

public class AutoCheck {
    public static final boolean isOnlineLited = false; // 是否只需要是纯在线识别功能
    private LinkedHashMap<String, Check> checks;

    private Context context;
    private Handler handler;

    private boolean hasError;
    private boolean enableOffline;
    private boolean isFinished = false;

    private String name;

    private static final String TAG = "AutoCheck";

    public AutoCheck(Context context, final Handler handler, boolean enableOffline) {
        this.context = context;
        checks = new LinkedHashMap<>();
        this.handler = handler;
        this.enableOffline = enableOffline;
    }

    public void checkAsr(final Map<String, Object> params) {
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                AutoCheck obj = checkAsrInternal(params);
                name = "识别";
                synchronized (obj) { // 偶发，同步线程信息
                    isFinished = true;
                    Message msg = handler.obtainMessage(100, obj);
                    handler.sendMessage(msg);
                }
            }
        });
        t.start();
    }

    public String obtainErrorMessage() {
        PrintConfig config = new PrintConfig();
        return formatString(config);
    }

    public String obtainDebugMessage() {
        PrintConfig config = new PrintConfig();
        config.withInfo = true;
        return formatString(config);
    }

    public String obtainAllMessage() {
        PrintConfig config = new PrintConfig();
        config.withLog = true;
        config.withInfo = true;
        config.withLogOnSuccess = true;
        return formatString(config);
    }

    private String formatString(PrintConfig config) {
        StringBuilder sb = new StringBuilder();
        hasError = false;

        for (HashMap.Entry<String, Check> entry : checks.entrySet()) {
            Check check = entry.getValue();
            String testName = entry.getKey();
            if (check.hasError()) {
                if (!hasError) {
                    hasError = true;
                }

                sb.append("【错误】【").append(testName).append(" 】  ").append(check.getErrorMessage()).append("\n");
                Log.e("AutoCheck", sb.toString());
                if (check.hasFix()) {
                    sb.append("【修复方法】【").append(testName).append(" 】  ").append(check.getFixMessage()).append("\n");
                }
            } else if (config.withEachCheckInfo) {
                sb.append("【无报错】【").append(testName).append(" 】  ").append("\n");
            }
            if (config.withInfo && check.hasInfo()) {
                sb.append("【请手动检查】【").append(testName).append("】 ").append(check.getInfoMessage()).append("\n");
            }
            if (config.withLog && (config.withLogOnSuccess || hasError) && check.hasLog()) {
                sb.append("【log】:" + check.getLogMessage()).append("\n");
            }
        }
        if (!hasError) {
            sb.append("【" + name + "】集成自动排查工具： 恭喜没有检测到任何问题\n");
        }
        return sb.toString();
    }

    private AutoCheck checkAsrInternal(Map<String, Object> params) {
        commonSetting(params);
        checks.put("外部音频文件存在校验", new FileCheck(context, params, SpeechConstant.IN_FILE));
        checks.put("离线命令词及本地语义bsg文件存在校验",
                new FileCheck(context, params, SpeechConstant.ASR_OFFLINE_ENGINE_GRAMMER_FILE_PATH));
        for (Map.Entry<String, Check> e : checks.entrySet()) {
            Check check = e.getValue();
            check.check();
            if (check.hasError()) {
                break;
            }
        }
        return this;
    }

    private void commonSetting(Map<String, Object> params) {
        checks.put("检查申请的Android权限", new PermissionCheck(context));
        checks.put("检查so文件是否存在", new JniCheck(context));
        AppInfoCheck infoCheck = null;
        try {
            infoCheck = new AppInfoCheck(context, params);
            checks.put("检查AppId AppKey SecretKey", infoCheck);
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
            Log.e(TAG, "检查AppId AppKey SecretKey 错误", e);
            return;
        }
        if (enableOffline) {
            checks.put("检查包名", new ApplicationIdCheck(context, infoCheck.appId));
        }

    }

    private static class PrintConfig {
        public boolean withEachCheckInfo = false;
        public boolean withInfo = false;
        public boolean withLog = false;
        public boolean withLogOnSuccess = false;
    }


    private static class PermissionCheck extends Check {
        private Context context;

        public PermissionCheck(Context context) {
            this.context = context;
        }

        @Override
        public void check() {
            String[] permissions = {
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.ACCESS_NETWORK_STATE,
                    Manifest.permission.INTERNET,
                    // Manifest.permission.WRITE_EXTERNAL_STORAGE,
            };

            ArrayList<String> toApplyList = new ArrayList<String>();
            for (String perm : permissions) {
                if (PackageManager.PERMISSION_GRANTED != ContextCompat.checkSelfPermission(context, perm)) {
                    toApplyList.add(perm);
                    // 进入到这里代表没有权限.
                }
            }
            if (!toApplyList.isEmpty()) {
                errorMessage = "缺少权限：" + toApplyList;
                fixMessage = "请从AndroidManifest.xml复制相关权限";
            }
        }
    }

    private static class JniCheck extends Check {
        private Context context;

        private String[] soNames;

        public JniCheck(Context context) {
            this.context = context;
            if (isOnlineLited) {
                soNames = new String[]{"libBaiduSpeechSDK.so", "libvad.dnn.so"};
            } else {
                soNames = new String[]{"libBaiduSpeechSDK.so", "libvad.dnn.so",
                        "libbd_easr_s1_merge_normal_20151216.dat.so", "libbdEASRAndroid.so",
                        "libbdSpilWakeup.so"};
            }
        }

        @Override
        public void check() {
            String path = context.getApplicationInfo().nativeLibraryDir;
            appendLogMessage("Jni so文件目录 " + path);
            File[] files = new File(path).listFiles();
            TreeSet<String> set = new TreeSet<>();
            if (files != null) {
                for (File file : files) {
                    set.add(file.getName());
                }
            }
            // String debugMessage = "Jni目录内文件: " + set.toString();
            // boolean isSuccess = true;
            for (String name : soNames) {
                if (!set.contains(name)) {
                    errorMessage = "Jni目录" + path + " 缺少so文件：" + name + "， 该目录文件列表: " + set.toString();
                    fixMessage = "如果您的app内没有其它so文件，请复制demo里的src/main/jniLibs至同名目录。"
                            + " 如果app内有so文件，请合并目录放一起(注意目录取交集，多余的目录删除)。";
                    break;
                }
            }
        }
    }

    private static class AppInfoCheck extends Check {
        private String appId;
        private String appKey;
        private String secretKey;

        public AppInfoCheck(Context context, Map<String, Object> params) throws PackageManager.NameNotFoundException {

            if (params.get(SpeechConstant.APP_ID) != null) {
                appId = params.get(SpeechConstant.APP_ID).toString();
            }
            if (params.get(SpeechConstant.APP_KEY) != null) {
                appKey = params.get(SpeechConstant.APP_KEY).toString();
            }

            if (params.get(SpeechConstant.SECRET) != null) {
                secretKey = params.get(SpeechConstant.SECRET).toString();
            }
        }


        public void check() {
            do {
                appendLogMessage("try to check appId " + appId + " ,appKey=" + appKey + " ,secretKey" + secretKey);
                if (appId == null || appId.isEmpty()) {
                    errorMessage = "appId 为空";
                    fixMessage = "填写appID";
                    break;
                }
                if (appKey == null || appKey.isEmpty()) {
                    errorMessage = "appKey 为空";
                    fixMessage = "填写appID";
                    break;
                }
                if (secretKey == null || secretKey.isEmpty()) {
                    errorMessage = "secretKey 为空";
                    fixMessage = "secretKey";
                    break;
                }


                try {
                    checkOnline();
                } catch (UnknownHostException e) {
                    infoMessage = "无网络或者网络不连通，忽略检测 : " + e.getMessage();
                } catch (Exception e) {
                    errorMessage = e.getClass().getCanonicalName() + ":" + e.getMessage();
                    fixMessage = " 重新检测appId， appKey， appSecret是否正确";
                }
            } while (false);
        }

        public void checkOnline() throws Exception {
            String urlpath = "https://openapi.baidu.com/oauth/2.0/token?client_id="
                    + appKey + "&client_secret=" + secretKey + "&grant_type=client_credentials";
            Log.i("AutoCheck", "Url is " + urlpath);
            URL url = new URL(urlpath);
            HttpsURLConnection conn = (HttpsURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(1000);
            InputStream is = conn.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            StringBuilder result = new StringBuilder();
            String line = "";
            do {
                line = reader.readLine();
                if (line != null) {
                    result.append(line);
                }
            } while (line != null);
            String res = result.toString();
            if (!res.contains("audio_voice_assistant_get")) {
                errorMessage = "appid:" + appId + ",没有audio_voice_assistant_get 权限，请在网页上开通\"语音识别\"能力";
                fixMessage = "secretKey";
                return;
            }
            appendLogMessage("openapi return " + res);
            JSONObject jsonObject = new JSONObject(res);
            String error = jsonObject.optString("error");
            if (error != null && !error.isEmpty()) {
                errorMessage = "appkey secretKey 错误" + ", error:" + error + ", json is" + result;
                fixMessage = " 重新检测appId对应的 appKey， appSecret是否正确";
                return;
            }
            String token = jsonObject.getString("access_token");
            if (token == null || !token.endsWith("-" + appId)) {
                errorMessage = "appId 与 appkey及 appSecret 不一致。appId = " + appId + " ,token = " + token;
                fixMessage = " 重新检测appId对应的 appKey， appSecret是否正确";
            }
        }
    }

    private static class ApplicationIdCheck extends Check {

        private String appId;
        private Context context;

        public ApplicationIdCheck(Context context, String appId) {
            this.appId = appId;
            this.context = context;
        }

        @Override
        public void check() {
            infoMessage = "如果您集成过程中遇见离线命令词或者唤醒初始化问题，请检查网页上appId：" + appId
                    + "  应用填写了Android包名："
                    + getApplicationId();
        }

        private String getApplicationId() {
            return context.getPackageName();
        }
    }

    private static class FileCheck extends Check {
        private Map<String, Object> params;
        private String key;
        private Context context;
        private boolean allowRes = false;
        private boolean allowAssets = true;

        public FileCheck(Context context, Map<String, Object> params, String key) {
            this.context = context;
            this.params = params;
            this.key = key;
            if (key.equals(SpeechConstant.IN_FILE)) {
                allowRes = true;
                allowAssets = false;
            }
        }

        @Override
        public void check() {
            if (!params.containsKey(key)) {
                return;
            }
            String value = params.get(key).toString();
            if (allowAssets) {
                int len = "assets".length();
                int totalLen = len + ":///".length();
                if (value.startsWith("assets")) {
                    String filename = value.substring(totalLen);
                    if (!":///".equals(value.substring(len, totalLen)) || filename.isEmpty()) {
                        errorMessage = "参数：" + key + "格式错误：" + value;
                        fixMessage = "修改成" + "assets:///sdcard/xxxx.yyy";
                    }
                    try {
                        context.getAssets().open(filename);
                    } catch (IOException e) {
                        errorMessage = "assets 目录下，文件不存在：" + filename;
                        fixMessage = "demo的assets目录是：src/main/assets";
                        e.printStackTrace();
                    }
                    appendLogMessage("assets 检验完毕：" + filename);
                }
            }
            if (allowRes) {
                int len = "res".length();
                int totalLen = len + ":///".length();
                if (value.startsWith("res")) {
                    String filename = value.substring(totalLen);
                    if (!":///".equals(value.substring(len, totalLen)) || filename.isEmpty()) {
                        errorMessage = "参数：" + key + "格式错误：" + value;
                        fixMessage = "修改成" + "res:///com/baidu/android/voicedemo/16k_test.pcm";
                    }
                    InputStream is = getClass().getClassLoader().getResourceAsStream(filename);
                    if (is == null) {
                        errorMessage = "res，文件不存在：" + filename;
                        fixMessage = "demo的res目录是：app/src/main/resources";
                    } else {
                        try {
                            is.close();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                    appendLogMessage("res 检验完毕：" + filename);
                }
            }
            if (value.startsWith("/")) {
                if (!new File(value).canRead()) {
                    errorMessage = "文件不存在：" + value;
                    fixMessage = "请查看文件是否存在";
                }
                appendLogMessage("文件路径 检验完毕：" + value);
            }
        }
    }

    private abstract static class Check {
        protected String errorMessage = null;

        protected String fixMessage = null;

        protected String infoMessage = null;

        protected StringBuilder logMessage;

        public Check() {
            logMessage = new StringBuilder();
        }

        public abstract void check();

        public boolean hasError() {
            return errorMessage != null;
        }

        public boolean hasFix() {
            return fixMessage != null;
        }

        public boolean hasInfo() {
            return infoMessage != null;
        }

        public boolean hasLog() {
            return !logMessage.toString().isEmpty();
        }

        public void appendLogMessage(String message) {
            logMessage.append(message + "\n");
        }

        public String getErrorMessage() {
            return errorMessage;
        }

        public String getFixMessage() {
            return fixMessage;
        }

        public String getInfoMessage() {
            return infoMessage;
        }

        public String getLogMessage() {
            return logMessage.toString();
        }
    }
}

