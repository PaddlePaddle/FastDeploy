package com.baidu.paddle.fastdeploy.app.ui.camera;

import android.view.TextureView;


/**
 * Created by ruanshimin on 2017/3/29.
 */
public interface ICameraProxy extends TextureView.SurfaceTextureListener {
    void openCamera();

    void setDisplayRotation(int degree);

    void startPreview();

    void stopPreview();

    void closeCamera();

    void switchSide();

    int[] getPreviewSize();

    void takePicture(CameraListener.TakePictureListener listener);

    void setEventListener(CameraListener.CommonListener commonListener);
}
