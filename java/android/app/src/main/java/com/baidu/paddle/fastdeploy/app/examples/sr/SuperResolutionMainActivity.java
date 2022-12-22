package com.baidu.paddle.fastdeploy.app.examples.sr;

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
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;

import com.baidu.paddle.fastdeploy.app.examples.R;
import com.baidu.paddle.fastdeploy.app.examples.text.applications.VoiceAssistantMainActivity;
import com.baidu.paddle.fastdeploy.app.examples.text.applications.VoiceAssistantSettingsActivity;
import com.davemorrissey.labs.subscaleview.ImageSource;
import com.davemorrissey.labs.subscaleview.SubsamplingScaleImageView;

import java.io.IOException;
import java.io.InputStream;

public class SuperResolutionMainActivity extends Activity implements View.OnClickListener, AdapterView.OnItemSelectedListener, View.OnTouchListener {
    private SubsamplingScaleImageView originImage;
    private SubsamplingScaleImageView resultImage;
    private ImageView back;
    private Bitmap defaultBitmap;
    private Bitmap selectBitmap;
    private Spinner spinner;
    private Button start;
    private Button reset;
    private ImageButton btnSettings;
    private ImageView albumSelect;
    private int multiple = 0;
    private boolean isAlbumSelect = false;
    private static final int REQUEST_PERMISSION_CODE_STORAGE = 101;
    private static final int INTENT_CODE_PICK_IMAGE = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Fullscreen
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.super_resolution_activity_main);

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
            case R.id.btn_start:
                if (multiple == 0) {
                    new AlertDialog.Builder(SuperResolutionMainActivity.this)
                            .setMessage("请先选择放大倍数。")
                            .setCancelable(true)
                            .show();
                    return;
                } else if (multiple == 1) {
                    if (isAlbumSelect) {
                        // TODO: 2022/12/15 接入算法
                    } else {
                        startMultiple(1, "super_pic_1.jpg");
                    }
                } else if (multiple == 2) {
                    if (isAlbumSelect) {
                        // TODO: 2022/12/15 接入算法
                    } else {
                        startMultiple(2, "super_pic_2.jpg");
                    }
                } else if (multiple == 3) {
                    if (isAlbumSelect) {
                        // TODO: 2022/12/15 接入算法
                    } else {
                        startMultiple(4, "super_pic_4.jpg");
                    }
                }
                break;
            case R.id.btn_settings:
                startActivity(new Intent(SuperResolutionMainActivity.this, SuperResolutionSettingsActivity.class));
                break;
            case R.id.btn_reset:
                originImage.resetScaleAndCenter();
                resultImage.resetScaleAndCenter();
                break;
            case R.id.album_select:
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
            case R.id.back_in_result:
                finish();
                break;
        }
    }

    private void startMultiple(int scale, String file) {
        originImage.resetScaleAndCenter();
        defaultBitmap = changeImageScale(2);
        originImage.setImage(ImageSource.bitmap(changeImageScale(scale, defaultBitmap)));
        resultImage.setImage(ImageSource.asset(file));
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == INTENT_CODE_PICK_IMAGE) {
            if (resultCode == Activity.RESULT_OK) {
                Uri uri = data.getData();
                String path = getRealPathFromURI(this, uri);
                selectBitmap = BitmapFactory.decodeFile(path);
                originImage.resetScaleAndCenter();
                originImage.setImage(ImageSource.bitmap(changeImageScale(1, selectBitmap)));
                isAlbumSelect = true;
                resultImage.recycle();
                spinner.setSelection(0);
            }
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

    private Bitmap changeImageScale(int scale, Bitmap bitmap) {
        Matrix matrix = new Matrix();
        matrix.postScale(scale, scale);
        Bitmap createBmp = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        return createBmp;
    }

    public void initView() {
        originImage = findViewById(R.id.origin_image);
        originImage.setOnTouchListener(this);
        originImage.resetScaleAndCenter();
        defaultBitmap = changeImageScale(2);
        originImage.setImage(ImageSource.bitmap(changeImageScale(1, defaultBitmap)));
        resultImage = findViewById(R.id.result_image);
        resultImage.setOnTouchListener(this);
        back = findViewById(R.id.back_in_result);
        back.setOnClickListener(this);
        spinner = findViewById(R.id.spinner);
        spinner.setOnItemSelectedListener(this);
        start = findViewById(R.id.btn_start);
        start.setOnClickListener(this);
        reset = findViewById(R.id.btn_reset);
        reset.setOnClickListener(this);
        albumSelect = findViewById(R.id.album_select);
        albumSelect.setOnClickListener(this);
        btnSettings = findViewById(R.id.btn_settings);
        btnSettings.setOnClickListener(this);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            new AlertDialog.Builder(SuperResolutionMainActivity.this)
                    .setTitle("Permission denied")
                    .setMessage("Click to force quit the app, then open Settings->Apps & notifications->Target " +
                            "App->Permissions to grant all of the permissions.")
                    .setCancelable(false)
                    .setPositiveButton("Exit", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            SuperResolutionMainActivity.this.finish();
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
