package com.baidu.paddle.fastdeploy.app.ui.camera;

import android.graphics.Bitmap;

/**
 * Created by ruanshimin on 2017/11/30.
 */

public class CameraListener {
    public interface CommonListener {
        void onSurfaceReady();

        void onSwitchCamera();
    }

    public interface TakePictureListener {
        void onTakenPicture(Bitmap bitmap);
    }
}
