package com.baidu.paddle.fastdeploy.app.examples.superscript;

import static com.baidu.paddle.fastdeploy.ui.Utils.getRealPathFromURI;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import com.baidu.paddle.fastdeploy.app.examples.R;
import com.baidu.paddle.fastdeploy.ui.view.CameraSurfaceView;
import com.davemorrissey.labs.subscaleview.ImageSource;
import com.davemorrissey.labs.subscaleview.SubsamplingScaleImageView;

import java.io.IOException;
import java.io.InputStream;

public class SuperscriptMainActivity extends Activity implements View.OnClickListener, CameraSurfaceView.OnTextureChangedListener, AdapterView.OnItemSelectedListener, View.OnTouchListener {
    CameraSurfaceView svPreview;
    TextView tvStatus;
    ImageButton btnSwitch;
    ImageButton btnShutter;
    private ImageView albumSelectButton;
    private View cameraPageView;
    private View resultPageView;
    private SubsamplingScaleImageView originImage;
    private SubsamplingScaleImageView resultImage;
    private ImageView backInResult;
    private ImageView backInPreview;
    private Bitmap shutterBitmap;
    private Bitmap selectBitmap;
    private Bitmap originBitmap;
    private Spinner spinner;
    private Button start;
    private Button reset;
    private static int BACK_TYPE;
    private static final int BACK_TYPE_RESULT = 1;
    private static final int BACK_TYPE_PREVIEW = 2;
    private boolean isShutterBitmapCopied = false;

    public static final int TYPE_UNKNOWN = -1;
    public static final int BTN_SHUTTER = 0;
    public static final int ALBUM_SELECT = 1;
    public static final int REALTIME_SUPER = 2;
    private static int TYPE = REALTIME_SUPER;
    private static final int REQUEST_PERMISSION_CODE_STORAGE = 101;
    private static final int INTENT_CODE_PICK_IMAGE = 100;
    private static final int TIME_SLEEP_INTERVAL = 50; // ms
    private int multiple = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Fullscreen
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.superscript_activity_main);

        // Check and request CAMERA and WRITE_EXTERNAL_STORAGE permissions
        if (!checkAllPermissions()) {
            requestAllPermissions();
        }

        // Init the camera preview and UI components
        initView();
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
                shutterAndPauseCamera();
                break;
            case R.id.album_select:
                TYPE = ALBUM_SELECT;
                // Judge whether authority has been granted.
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                    // If this permission was requested before the application but the user refused the request, this method will return true.
                    ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_PERMISSION_CODE_STORAGE);
                } else {
                    Intent intent = new Intent(Intent.ACTION_PICK);
                    intent.setType("image/*");
                    startActivityForResult(intent, INTENT_CODE_PICK_IMAGE);
                }
                break;
            case R.id.btn_start:
                if (multiple == 0) {
                    new AlertDialog.Builder(SuperscriptMainActivity.this)
                            .setMessage("请先选择放大倍数。")
                            .setCancelable(true)
                            .show();
                    return;
                } else if (multiple == 1) {
                    originImage.resetScaleAndCenter();
                    originBitmap = changeImageScale(2);
                    originImage.setImage(ImageSource.bitmap(changeImageScale(1, originBitmap)));
                    resultImage.setImage(ImageSource.asset("super_pic_1.jpg"));
                } else if (multiple == 2) {
                    originImage.resetScaleAndCenter();
                    originBitmap = changeImageScale(2);
                    originImage.setImage(ImageSource.bitmap(changeImageScale(2, originBitmap)));
                    resultImage.setImage(ImageSource.asset("super_pic_2.jpg"));
                } else if (multiple == 3) {
                    originImage.resetScaleAndCenter();
                    originBitmap = changeImageScale(2);
                    originImage.setImage(ImageSource.bitmap(changeImageScale(4, originBitmap)));
                    resultImage.setImage(ImageSource.asset("super_pic_4.jpg"));
                }
                break;
            case R.id.btn_reset:
                originImage.resetScaleAndCenter();
                resultImage.resetScaleAndCenter();
                break;
            case R.id.back_in_result:
                backResult();
                break;
            case R.id.back_in_preview:
                finish();
                break;
        }
    }

    private Bitmap changeImageScale(int scale) {
        try {
            AssetManager am = this.getAssets();
            InputStream inputStream = null;
            inputStream = am.open("super_pic_-2.jpg");
            Matrix matrix = new Matrix();
            matrix.postScale(scale, scale);
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
            Bitmap createBmp = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            return createBmp;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public void onBackPressed() {
        if (BACK_TYPE == BACK_TYPE_RESULT) {
            backResult();
        } else {
            finish();
        }
    }

    private Bitmap changeImageScale(int scale, Bitmap bitmap) {
        Matrix matrix = new Matrix();
        matrix.postScale(scale, scale);
        Bitmap createBmp = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        return createBmp;
    }

    private void backResult() {
        originImage.recycle();
        resultImage.recycle();
        spinner.setSelection(0);
        BACK_TYPE = BACK_TYPE_PREVIEW;
        resultPageView.setVisibility(View.GONE);
        cameraPageView.setVisibility(View.VISIBLE);
        TYPE = REALTIME_SUPER;
        isShutterBitmapCopied = false;
        svPreview.onResume();
    }

    private void shutterAndPauseCamera() {
        BACK_TYPE = BACK_TYPE_RESULT;
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // Sleep some times to ensure picture has been correctly shut.
                    Thread.sleep(TIME_SLEEP_INTERVAL * 10); // 500ms
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                runOnUiThread(new Runnable() {
                    @SuppressLint("SetTextI18n")
                    public void run() {
                        // These codes will run in main thread.
                        svPreview.onPause();
                        cameraPageView.setVisibility(View.GONE);
                        resultPageView.setVisibility(View.VISIBLE);
                        if (shutterBitmap != null && !shutterBitmap.isRecycled()) {
                            //originImage.setImage(ImageSource.bitmap(shutterBitmap));
                            originBitmap = changeImageScale(2);
                            originImage.setImage(ImageSource.bitmap(originBitmap));
                        } else {
                            new AlertDialog.Builder(SuperscriptMainActivity.this)
                                    .setTitle("Empty Result!")
                                    .setMessage("Current picture is empty, please shutting it again!")
                                    .setCancelable(true)
                                    .show();
                        }
                    }
                });
            }
        }).start();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        BACK_TYPE = BACK_TYPE_RESULT;
        if (requestCode == INTENT_CODE_PICK_IMAGE) {
            if (resultCode == Activity.RESULT_OK) {
                cameraPageView.setVisibility(View.GONE);
                resultPageView.setVisibility(View.VISIBLE);
                Uri uri = data.getData();
                String path = getRealPathFromURI(this, uri);
                Bitmap bitmap = BitmapFactory.decodeFile(path);
                selectBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                SystemClock.sleep(TIME_SLEEP_INTERVAL * 10); // 500ms
                //originImage.setImage(ImageSource.bitmap(selectBitmap));
                originBitmap = changeImageScale(2);
                originImage.setImage(ImageSource.bitmap(originBitmap));
            }
        }
    }

    private void copyBitmapFromCamera(Bitmap ARGB8888ImageBitmap) {
        if (isShutterBitmapCopied || ARGB8888ImageBitmap == null) {
            return;
        }
        if (!ARGB8888ImageBitmap.isRecycled()) {
            synchronized (this) {
                shutterBitmap = ARGB8888ImageBitmap.copy(Bitmap.Config.ARGB_8888, true);
            }
            SystemClock.sleep(TIME_SLEEP_INTERVAL);
            isShutterBitmapCopied = true;
        }
    }

    @Override
    public boolean onTextureChanged(Bitmap ARGB8888ImageBitmap) {
        if (TYPE == BTN_SHUTTER) {
            copyBitmapFromCamera(ARGB8888ImageBitmap);
            return false;
        }
        return true;
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Open camera until the permissions have been granted
        if (!checkAllPermissions()) {
            svPreview.disableCamera();
        } else {
            svPreview.enableCamera();
        }
        svPreview.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
        svPreview.onPause();
    }

    public void initView() {
        TYPE = REALTIME_SUPER;
        CameraSurfaceView.EXPECTED_PREVIEW_WIDTH = 480;
        CameraSurfaceView.EXPECTED_PREVIEW_HEIGHT = 480;
        svPreview = (CameraSurfaceView) findViewById(R.id.sv_preview);
        svPreview.setOnTextureChangedListener(this);
        tvStatus = (TextView) findViewById(R.id.tv_status);
        btnSwitch = (ImageButton) findViewById(R.id.btn_switch);
        btnSwitch.setOnClickListener(this);
        btnShutter = (ImageButton) findViewById(R.id.btn_shutter);
        btnShutter.setOnClickListener(this);
        backInPreview = findViewById(R.id.back_in_preview);
        backInPreview.setOnClickListener(this);
        albumSelectButton = findViewById(R.id.album_select);
        albumSelectButton.setOnClickListener(this);
        cameraPageView = findViewById(R.id.camera_page);
        resultPageView = findViewById(R.id.result_page);
        originImage = findViewById(R.id.origin_image);
        originImage.setOnTouchListener(this);
        resultImage = findViewById(R.id.result_image);
        resultImage.setOnTouchListener(this);
        backInResult = findViewById(R.id.back_in_result);
        backInResult.setOnClickListener(this);
        spinner = findViewById(R.id.spinner);
        spinner.setOnItemSelectedListener(this);
        start = findViewById(R.id.btn_start);
        start.setOnClickListener(this);
        reset = findViewById(R.id.btn_reset);
        reset.setOnClickListener(this);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            new AlertDialog.Builder(SuperscriptMainActivity.this)
                    .setTitle("Permission denied")
                    .setMessage("Click to force quit the app, then open Settings->Apps & notifications->Target " +
                            "App->Permissions to grant all of the permissions.")
                    .setCancelable(false)
                    .setPositiveButton("Exit", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            SuperscriptMainActivity.this.finish();
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

    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
        multiple = i;
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {

    }

    @Override
    public boolean onTouch(View view, MotionEvent motionEvent) {
        switch (view.getId()) {
            case R.id.origin_image:
                resultImage.onTouchEvent(motionEvent);
                break;
            case R.id.result_image:
                originImage.onTouchEvent(motionEvent);
                break;
        }
        return false;
    }
}
