package com.baidu.paddle.fastdeploy.app.examples.detection;


import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.app.examples.R;
import com.baidu.paddle.fastdeploy.app.ui.view.CameraSurfaceView;
import com.baidu.paddle.fastdeploy.app.ui.view.ResultListView;
import com.baidu.paddle.fastdeploy.app.ui.Utils;
import com.baidu.paddle.fastdeploy.app.ui.view.adapter.DetectResultAdapter;
import com.baidu.paddle.fastdeploy.app.ui.view.model.BaseResultModel;
import com.baidu.paddle.fastdeploy.vision.DetectionResult;
import com.baidu.paddle.fastdeploy.vision.detection.PicoDet;

import java.io.File;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class DetectionMainActivity extends Activity implements View.OnClickListener, CameraSurfaceView.OnTextureChangedListener {
    private static final String TAG = DetectionMainActivity.class.getSimpleName();

    CameraSurfaceView svPreview;
    TextView tvStatus;
    ImageButton btnSwitch;
    ImageButton btnShutter;
    ImageButton btnSettings;
    ImageView realtimeToggleButton;
    boolean isRealtimeStatusRunning = false;
    ImageView backInPreview;
    private ImageView albumSelectButton;
    private View mCameraPageView;
    private ViewGroup mResultPageView;
    private ImageView resultImage;
    private ImageView backInResult;
    private SeekBar confidenceSeekbar;
    private TextView seekbarText;
    private float resultNum = 1.0f;
    private ResultListView detectResultView;
    private Bitmap shutterBitmap;
    private Bitmap originShutterBitmap;
    private Bitmap picBitmap;
    private Bitmap originPicBitmap;
    public static final int BTN_SHUTTER = 0;
    public static final int ALBUM_SELECT = 1;
    private static int TYPE = BTN_SHUTTER;

    private static final int REQUEST_PERMISSION_CODE_STORAGE = 101;
    private static final int INTENT_CODE_PICK_IMAGE = 100;

    String savedImagePath = "result.jpg";
    int lastFrameIndex = 0;
    long lastFrameTime;

    // Call 'init' and 'release' manually later
    PicoDet predictor = new PicoDet();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Fullscreen
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.detection_activity_main);

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
                TYPE = BTN_SHUTTER;
                svPreview.onPause();
                mCameraPageView.setVisibility(View.GONE);
                mResultPageView.setVisibility(View.VISIBLE);
                seekbarText.setText(resultNum + "");
                confidenceSeekbar.setProgress((int) (resultNum * 100));
                resultImage.setImageBitmap(shutterBitmap);
                break;
            case R.id.btn_settings:
                startActivity(new Intent(DetectionMainActivity.this, DetectionSettingsActivity.class));
                break;
            case R.id.realtime_toggle_btn:
                toggleRealtimeStyle();
                break;
            case R.id.back_in_preview:
                finish();
                break;
            case R.id.albumSelect:
                TYPE = ALBUM_SELECT;
                // 判断是否已经赋予权限
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                    // 如果应用之前请求过此权限但用户拒绝了请求，此方法将返回 true。
                    ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_PERMISSION_CODE_STORAGE);
                } else {
                    Intent intent = new Intent(Intent.ACTION_PICK);
                    intent.setType("image/*");
                    startActivityForResult(intent, INTENT_CODE_PICK_IMAGE);
                }
                break;
            case R.id.back_in_result:
                mResultPageView.setVisibility(View.GONE);
                mCameraPageView.setVisibility(View.VISIBLE);
                svPreview.onResume();
                break;

        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == INTENT_CODE_PICK_IMAGE) {
            if (resultCode == Activity.RESULT_OK) {
                mCameraPageView.setVisibility(View.GONE);
                mResultPageView.setVisibility(View.VISIBLE);
                seekbarText.setText(resultNum + "");
                confidenceSeekbar.setProgress((int) (resultNum * 100));
                Uri uri = data.getData();
                String path = getRealPathFromURI(uri);
                picBitmap = decodeBitmap(path, 720, 1280);
                originPicBitmap = picBitmap.copy(Bitmap.Config.ARGB_8888, true);
                resultImage.setImageBitmap(picBitmap);
            }
        }
    }

    private String getRealPathFromURI(Uri contentURI) {
        String result;
        Cursor cursor = null;
        try {
            cursor = getContentResolver().query(contentURI, null, null, null, null);
        } catch (Throwable e) {
            e.printStackTrace();
        }
        if (cursor == null) {
            result = contentURI.getPath();
        } else {
            cursor.moveToFirst();
            int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
            result = cursor.getString(idx);
            cursor.close();
        }
        return result;
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
            savedImagePath = Utils.getDCIMDirectory() + File.separator + "result.png";
        }
        shutterBitmap = ARGB8888ImageBitmap.copy(Bitmap.Config.ARGB_8888,true);
        originShutterBitmap = ARGB8888ImageBitmap.copy(Bitmap.Config.ARGB_8888,true);
        boolean modified = false;
        DetectionResult result = predictor.predict(
                ARGB8888ImageBitmap, savedImagePath, DetectionSettingsActivity.scoreThreshold);
        modified = result.initialized();
        if (!savedImagePath.isEmpty()) {
            synchronized (this) {
                DetectionMainActivity.this.savedImagePath = "result.jpg";
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

    /**
     * @param path          路径
     * @param displayWidth  需要显示的宽度
     * @param displayHeight 需要显示的高度
     * @return Bitmap
     */
    public static Bitmap decodeBitmap(String path, int displayWidth, int displayHeight) {
        BitmapFactory.Options op = new BitmapFactory.Options();
        op.inJustDecodeBounds = true;
        // op.inJustDecodeBounds = true;表示我们只读取Bitmap的宽高等信息，不读取像素。
        Bitmap bmp = BitmapFactory.decodeFile(path, op); // 获取尺寸信息
        // op.outWidth表示的是图像真实的宽度
        // op.inSamplySize 表示的是缩小的比例
        // op.inSamplySize = 4,表示缩小1/4的宽和高，1/16的像素，android认为设置为2是最快的。
        // 获取比例大小
        int wRatio = (int) Math.ceil(op.outWidth / (float) displayWidth);
        int hRatio = (int) Math.ceil(op.outHeight / (float) displayHeight);
        // 如果超出指定大小，则缩小相应的比例
        if (wRatio > 1 && hRatio > 1) {
            if (wRatio > hRatio) {
                // 如果太宽，我们就缩小宽度到需要的大小，注意，高度就会变得更加的小。
                op.inSampleSize = wRatio;
            } else {
                op.inSampleSize = hRatio;
            }
        }
        op.inJustDecodeBounds = false;
        bmp = BitmapFactory.decodeFile(path, op);
        // 从原Bitmap创建一个给定宽高的Bitmap
        return Bitmap.createScaledBitmap(bmp, displayWidth, displayHeight, true);
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
        albumSelectButton = findViewById(R.id.albumSelect);
        albumSelectButton.setOnClickListener(this);
        mCameraPageView = findViewById(R.id.camera_page);
        mResultPageView = findViewById(R.id.result_page);
        resultImage = findViewById(R.id.result_image);
        backInResult = findViewById(R.id.back_in_result);
        backInResult.setOnClickListener(this);
        confidenceSeekbar = findViewById(R.id.confidence_seekbar);
        seekbarText = findViewById(R.id.seekbar_text);
        detectResultView = findViewById(R.id.result_list_view);

        List<BaseResultModel> results = new ArrayList<>();
        results.add(new BaseResultModel(1, "cup", 0.4f));
        results.add(new BaseResultModel(2, "pen", 0.6f));
        results.add(new BaseResultModel(3, "tang", 1.0f));
        final DetectResultAdapter adapter = new DetectResultAdapter(this, R.layout.detection_result_page_item, results);
        detectResultView.setAdapter(adapter);
        detectResultView.invalidate();

        confidenceSeekbar.setMax(100);
        confidenceSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float resultConfidence = seekBar.getProgress() / 100f;
                BigDecimal bd = new BigDecimal(resultConfidence);
                resultNum = bd.setScale(1, BigDecimal.ROUND_HALF_UP).floatValue();
                seekbarText.setText(resultNum + "");
                confidenceSeekbar.setProgress((int) (resultNum * 100));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (TYPE == ALBUM_SELECT) {
                            SystemClock.sleep(500);
                            predictor.predict(picBitmap, savedImagePath, resultNum);
                            resultImage.setImageBitmap(picBitmap);
                            picBitmap = originPicBitmap.copy(Bitmap.Config.ARGB_8888, true);
                            resultNum = 1.0f;
                        } else {
                            SystemClock.sleep(500);
                            predictor.predict(shutterBitmap, savedImagePath, resultNum);
                            resultImage.setImageBitmap(shutterBitmap);
                            shutterBitmap = originShutterBitmap.copy(Bitmap.Config.ARGB_8888, true);
                            resultNum = 1.0f;
                        }
                    }
                });
            }
        });
    }

    @SuppressLint("ApplySharedPref")
    public void initSettings() {
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.clear();
        editor.commit();
        DetectionSettingsActivity.resetSettings();
    }

    public void checkAndUpdateSettings() {
        if (DetectionSettingsActivity.checkAndUpdateSettings(this)) {
            String realModelDir = getCacheDir() + "/" + DetectionSettingsActivity.modelDir;
            Utils.copyDirectoryFromAssets(this, DetectionSettingsActivity.modelDir, realModelDir);
            String realLabelPath = getCacheDir() + "/" + DetectionSettingsActivity.labelPath;
            Utils.copyFileFromAssets(this, DetectionSettingsActivity.labelPath, realLabelPath);

            String modelFile = realModelDir + "/" + "model.pdmodel";
            String paramsFile = realModelDir + "/" + "model.pdiparams";
            String configFile = realModelDir + "/" + "infer_cfg.yml";
            String labelFile = realLabelPath;
            RuntimeOption option = new RuntimeOption();
            option.setCpuThreadNum(DetectionSettingsActivity.cpuThreadNum);
            option.setLitePowerMode(DetectionSettingsActivity.cpuPowerMode);
            option.enableRecordTimeOfRuntime();
            if (Boolean.parseBoolean(DetectionSettingsActivity.enableLiteFp16)) {
                option.enableLiteFp16();
            }
            predictor.init(modelFile, paramsFile, configFile, labelFile, option);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            new AlertDialog.Builder(DetectionMainActivity.this)
                    .setTitle("Permission denied")
                    .setMessage("Click to force quit the app, then open Settings->Apps & notifications->Target " +
                            "App->Permissions to grant all of the permissions.")
                    .setCancelable(false)
                    .setPositiveButton("Exit", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            DetectionMainActivity.this.finish();
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
