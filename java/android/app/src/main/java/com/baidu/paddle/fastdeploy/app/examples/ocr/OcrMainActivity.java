package com.baidu.paddle.fastdeploy.app.examples.ocr;


import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.app.examples.R;
import com.baidu.paddle.fastdeploy.app.ui.Utils;
import com.baidu.paddle.fastdeploy.app.ui.view.CameraSurfaceView;
import com.baidu.paddle.fastdeploy.vision.OCRResult;
import com.baidu.paddle.fastdeploy.pipeline.PPOCRv2;
import com.baidu.paddle.fastdeploy.vision.ocr.Classifier;
import com.baidu.paddle.fastdeploy.vision.ocr.DBDetector;
import com.baidu.paddle.fastdeploy.vision.ocr.Recognizer;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

public class OcrMainActivity extends Activity implements View.OnClickListener, CameraSurfaceView.OnTextureChangedListener {
    private static final String TAG = OcrMainActivity.class.getSimpleName();

    CameraSurfaceView svPreview;
    TextView tvStatus;
    ImageButton btnSwitch;
    ImageButton btnShutter;
    ImageButton btnSettings;
    ImageView realtimeToggleButton;
    boolean isRealtimeStatusRunning = false;
    ImageView backInPreview;

    String savedImagePath = "result.jpg";
    int lastFrameIndex = 0;
    long lastFrameTime;

    // Call 'init' and 'release' manually later
    PPOCRv2 predictor = new PPOCRv2();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Fullscreen
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.ocr_activity_main);

        // Clear all setting items to avoid app crashing due to the incorrect settings
        initSettings();

        // Init the camera preview and UI components
        initView();

        // Check and request CAMERA and WRITE_EXTERNAL_STORAGE permissions
        if (!checkAllPermissions()) {
            requestAllPermissions();
        }
    }

    @SuppressLint("NonConstantResourceId")
    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.btn_switch:
                svPreview.switchCamera();
                break;
            case R.id.btn_shutter:
                @SuppressLint("SimpleDateFormat")
                SimpleDateFormat date = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
                synchronized (this) {
                    savedImagePath = Utils.getDCIMDirectory() + File.separator + date.format(new Date()).toString() + ".png";
                }
                Toast.makeText(OcrMainActivity.this, "Save snapshot to " + savedImagePath, Toast.LENGTH_SHORT).show();
                break;
            case R.id.btn_settings:
                startActivity(new Intent(OcrMainActivity.this, OcrSettingsActivity.class));
                break;
            case R.id.realtime_toggle_btn:
                toggleRealtimeStyle();
                break;
            case R.id.back_in_preview:
                finish();
                break;
        }
    }

    private void toggleRealtimeStyle() {
        if (isRealtimeStatusRunning) {
            isRealtimeStatusRunning = false;
            realtimeToggleButton.setImageResource(R.drawable.realtime_stop_btn);
            svPreview.setOnTextureChangedListener(this);
            tvStatus.setVisibility(View.VISIBLE);
        } else {
            isRealtimeStatusRunning = true;
            realtimeToggleButton.setImageResource(R.drawable.realtime_start_btn);
            tvStatus.setVisibility(View.GONE);
            svPreview.setOnTextureChangedListener(new CameraSurfaceView.OnTextureChangedListener() {
                @Override
                public boolean onTextureChanged(Bitmap ARGB8888ImageBitmap) {
                    return false;
                }
            });
        }
    }

    @Override
    public boolean onTextureChanged(Bitmap ARGB8888ImageBitmap) {
        String savedImagePath = "";
        synchronized (this) {
            savedImagePath = OcrMainActivity.this.savedImagePath;
        }
        boolean modified = false;
        OCRResult result = predictor.predict(ARGB8888ImageBitmap, savedImagePath);
        modified = result.initialized();
        if (!savedImagePath.isEmpty()) {
            synchronized (this) {
                OcrMainActivity.this.savedImagePath = "result.jpg";
            }
        }
        lastFrameIndex++;
        if (lastFrameIndex >= 30) {
            final int fps = (int) (lastFrameIndex * 1e9 / (System.nanoTime() - lastFrameTime));
            runOnUiThread(new Runnable() {
                @SuppressLint("SetTextI18n")
                public void run() {
                    tvStatus.setText(Integer.toString(fps) + "fps");
                }
            });
            lastFrameIndex = 0;
            lastFrameTime = System.nanoTime();
        }
        return modified;
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Reload settings and re-initialize the predictor
        checkAndUpdateSettings();
        // Open camera until the permissions have been granted
        if (!checkAllPermissions()) {
            svPreview.disableCamera();
        }
        svPreview.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
        svPreview.onPause();
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.release();
        }
        super.onDestroy();
    }

    public void initView() {
        svPreview = (CameraSurfaceView) findViewById(R.id.sv_preview);
        svPreview.setOnTextureChangedListener(this);
        tvStatus = (TextView) findViewById(R.id.tv_status);
        btnSwitch = (ImageButton) findViewById(R.id.btn_switch);
        btnSwitch.setOnClickListener(this);
        btnShutter = (ImageButton) findViewById(R.id.btn_shutter);
        btnShutter.setOnClickListener(this);
        btnSettings = (ImageButton) findViewById(R.id.btn_settings);
        btnSettings.setOnClickListener(this);
        realtimeToggleButton = findViewById(R.id.realtime_toggle_btn);
        realtimeToggleButton.setOnClickListener(this);
        backInPreview = findViewById(R.id.back_in_preview);
        backInPreview.setOnClickListener(this);
    }

    @SuppressLint("ApplySharedPref")
    public void initSettings() {
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.clear();
        editor.commit();
        OcrSettingsActivity.resetSettings();
    }

    public void checkAndUpdateSettings() {
        if (OcrSettingsActivity.checkAndUpdateSettings(this)) {
            String realModelDir = getCacheDir() + "/" + OcrSettingsActivity.modelDir;
            // String detModelName = "ch_PP-OCRv2_det_infer";
            String detModelName = "ch_PP-OCRv3_det_infer";
            // String detModelName = "ch_ppocr_mobile_v2.0_det_infer";
            String clsModelName = "ch_ppocr_mobile_v2.0_cls_infer";
            // String recModelName = "ch_ppocr_mobile_v2.0_rec_infer";
            String recModelName = "ch_PP-OCRv3_rec_infer";
            // String recModelName = "ch_PP-OCRv2_rec_infer";
            String realDetModelDir = realModelDir + "/" + detModelName;
            String realClsModelDir = realModelDir + "/" + clsModelName;
            String realRecModelDir = realModelDir + "/" + recModelName;
            String srcDetModelDir =  OcrSettingsActivity.modelDir + "/" + detModelName;
            String srcClsModelDir =  OcrSettingsActivity.modelDir + "/" + clsModelName;
            String srcRecModelDir =  OcrSettingsActivity.modelDir + "/" + recModelName;
            Utils.copyDirectoryFromAssets(this, srcDetModelDir, realDetModelDir);
            Utils.copyDirectoryFromAssets(this, srcClsModelDir, realClsModelDir);
            Utils.copyDirectoryFromAssets(this, srcRecModelDir, realRecModelDir);
            String realLabelPath = getCacheDir() + "/" + OcrSettingsActivity.labelPath;
            Utils.copyFileFromAssets(this, OcrSettingsActivity.labelPath, realLabelPath);

            String detModelFile = realDetModelDir + "/" + "inference.pdmodel";
            String detParamsFile = realDetModelDir + "/" + "inference.pdiparams";
            String clsModelFile = realClsModelDir + "/" + "inference.pdmodel";
            String clsParamsFile = realClsModelDir + "/" + "inference.pdiparams";
            String recModelFile = realRecModelDir + "/" + "inference.pdmodel";
            String recParamsFile = realRecModelDir + "/" + "inference.pdiparams";
            String recLabelFilePath = realLabelPath; // ppocr_keys_v1.txt
            RuntimeOption detOption = new RuntimeOption();
            RuntimeOption clsOption = new RuntimeOption();
            RuntimeOption recOption = new RuntimeOption();
            detOption.setCpuThreadNum(OcrSettingsActivity.cpuThreadNum);
            clsOption.setCpuThreadNum(OcrSettingsActivity.cpuThreadNum);
            recOption.setCpuThreadNum(OcrSettingsActivity.cpuThreadNum);
            detOption.setLitePowerMode(OcrSettingsActivity.cpuPowerMode);
            clsOption.setLitePowerMode(OcrSettingsActivity.cpuPowerMode);
            recOption.setLitePowerMode(OcrSettingsActivity.cpuPowerMode);
            detOption.enableRecordTimeOfRuntime();
            clsOption.enableRecordTimeOfRuntime();
            recOption.enableRecordTimeOfRuntime();
            if (Boolean.parseBoolean(OcrSettingsActivity.enableLiteFp16)) {
                detOption.enableLiteFp16();
                clsOption.enableLiteFp16();
                recOption.enableLiteFp16();
            }
            DBDetector detModel = new DBDetector(detModelFile, detParamsFile, detOption);
            Classifier clsModel = new Classifier(clsModelFile, clsParamsFile, clsOption);
            Recognizer recModel = new Recognizer(recModelFile, recParamsFile, recLabelFilePath, recOption);
            predictor.init(detModel, clsModel, recModel);

        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            new AlertDialog.Builder(OcrMainActivity.this)
                    .setTitle("Permission denied")
                    .setMessage("Click to force quit the app, then open Settings->Apps & notifications->Target " +
                            "App->Permissions to grant all of the permissions.")
                    .setCancelable(false)
                    .setPositiveButton("Exit", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            OcrMainActivity.this.finish();
                        }
                    }).show();
        }
    }

    private void requestAllPermissions() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.CAMERA}, 0);
    }

    private boolean checkAllPermissions() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }
}
